import asyncio
import time
import numpy as np
import mlx.core as mx

from events import EventType

from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)

INIT_PROMPT = """You are a voice assistant named Zoe. You hear short chunks of audio with possibly small overlaps.
Follow these rules:
* The audio transcription might contain errors, so guess what the user is saying.
* Generate a response only if needed.
* Prefer short responses
* Include commas in your responses.
* Use a spoken style response rather than a written style response, use commas to indicate a break in speech.

"""

INIT_EVENTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the time?", "Sorry, I don't know."),
    ("What is the capital of Germany?", "Berlin"),
    ("Stop", ""),
    ("How long would it take to", ""),
    ("How long would it take to walk from Paris to London", "Sorry, I don't know."),
    ("What is your name?", "Zoe")
]

def tokenize_initial_prompt(llm_tokenizer):
    tokens = [llm_tokenizer.bos_token_id]
    tokens += llm_tokenizer.encode("[INST]", add_special_tokens=False)
    tokens += llm_tokenizer.encode(INIT_PROMPT, add_special_tokens=False)
    tokens += llm_tokenizer.encode(INIT_EVENTS[0][0], add_special_tokens=False)
    tokens += llm_tokenizer.encode("[/INST]", add_special_tokens=False)
    tokens += llm_tokenizer.encode(INIT_EVENTS[0][1], add_special_tokens=False)
    tokens.append(llm_tokenizer.eos_token_id)

    for heard, response in INIT_EVENTS[1:]:
        tokens += llm_tokenizer.encode("[INST]", add_special_tokens=False)
        tokens += llm_tokenizer.encode(heard, add_special_tokens=False)
        tokens += llm_tokenizer.encode("[/INST]", add_special_tokens=False)
        tokens += llm_tokenizer.encode(response, add_special_tokens=False)
        tokens.append(llm_tokenizer.eos_token_id)

    # Start an initial user message segment
    tokens += llm_tokenizer.encode("[INST]", add_special_tokens=False)

    return tokens


@dataclass
class VoiceChatAgentConfig:
    num_tokens_response_pause: int = 10

class VoiceChatAgent:

    def __init__(self, pubsub, model, tokenizer, config):
        self.pubsub = pubsub
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.events_log = []
        self.new_event = asyncio.Event()

        self.pubsub.subscribe(EventType.HEARD_SPEECH, self.handle_heard_speech)

    def _model_call(self, token, llm_cache):
        # token_str = str(token.item())
        # with open("tokens_log.txt", "a") as file:
        #     file.write(token_str + "\n")
        #     file.flush()
        return self.model(token.reshape(1, 1), llm_cache)

    def _init_llm_cache(self):
        tokens = tokenize_initial_prompt(self.tokenizer)
        _, llm_cache = self.model(mx.array(tokens)[None])
        mx.eval(llm_cache)
        logger.debug("Initialized cache for LLM")
        return llm_cache

    def _update_heard_segment(self, transcribed_text, state, llm_cache):
        logger.debug(f"Updating LLM cache with: {transcribed_text}")
        for t in mx.array(self.tokenizer.encode(transcribed_text, add_special_tokens=False)):
            _, llm_cache = self._model_call(t, llm_cache)

        mx.eval(llm_cache)
        logger.debug("Updated LLM cache")

    def _generate_response_segment(self, state, response_tokens_so_far, next_response_token, llm_cache):
        if state != "RESPONDING":
            logger.debug(f"In state {state}, switching to RESPONDING, starting new response segment")
            response_tokens_so_far = []
            # for t in self._response_tokens:
            #     logits, llm_cache = self._model_call(t, llm_cache)
            for t in mx.array(self.tokenizer.encode("[/INST]", add_special_tokens=False)):
                logits, llm_cache = self._model_call(t, llm_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            logger.debug("Added response tokens")

        assert next_response_token is not None, "expected next_response_token to be initialized in responding state"
        
        say_response_so_far = False
        is_response_finished = False
        found_punctuation = False
        for _ in range(self.config.num_tokens_response_pause): 
            # newline indicates response is finished
            # if '\n' in decoded_next_token:
            if next_response_token.item() == self.tokenizer.eos_token_id:
                is_response_finished = True
                logger.debug("Found newline, breaking response generation")
                break

            response_tokens_so_far.append(next_response_token.item())
            decoded_next_token = self.tokenizer.decode([next_response_token.item()])

            # also check for any punctuation, as that indicates end of a sentence, and we
            # want to say each sentence asap
            if any(punct in decoded_next_token for punct in ',.!?'):
                found_punctuation = True
                logger.debug("Found punctuation, breaking response generation")
                break

            logits, llm_cache = self._model_call(next_response_token, llm_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)

        if is_response_finished:
            logger.debug("Finished response")
            # TODO: Trim off anything after last '\n' character, in case LLM generated next HEARD segment
            # TODO: actually this is a challenge as also need to fix cache to remove partial next segment
            say_response_so_far = True
            return response_tokens_so_far, next_response_token, say_response_so_far, is_response_finished

        elif found_punctuation:
            logger.debug("Found punctuation indicating end of a sentence")
            
            # Generate a new next token to avoid breaking on next iteration
            logits, llm_cache = self._model_call(next_response_token, llm_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            mx.eval(next_response_token)

            say_response_so_far = True
            return response_tokens_so_far, next_response_token, say_response_so_far, is_response_finished

        else:
            # else we are just pausing response to check for new audio,
            # and will keep generating if there are no interuptions
            logger.debug(f"Pausing to check for interuptions with partial response: {self.tokenizer.decode(response_tokens_so_far)}")
            mx.eval(next_response_token)
            return response_tokens_so_far, next_response_token, say_response_so_far, is_response_finished
        
    async def handle_heard_speech(self, transcript):
        logger.debug(f"LLM handling heard speech: {transcript}")
        # Remove any speech pause events, as they are no longer relevant
        self.events_log = [event for event in self.events_log if event[0] != EventType.HEARD_SPEECH_PAUSE]
        self.events_log.append([EventType.HEARD_SPEECH, transcript])
        self.new_event.set()

    async def run(self):
        state = "WAITING"
        next_response_token = None
        response_tokens_so_far = []
        llm_cache = self._init_llm_cache()

        while True:
            logger.debug(f"Entering main loop - state {state}")

            if not self.events_log:
                logger.debug("Waiting for event")
                self.new_event.clear()
                await self.new_event.wait()
                logger.debug("Event received")

            event_type, data = self.events_log.pop(0)
            assert event_type == EventType.HEARD_SPEECH, "expected only HEARD_SPEECH events"
            logger.info(f"Processing {event_type} event: {data}")

            self._update_heard_segment(data, state, llm_cache)
            state = "LISTENING"
            
            # Allow potenital interuption to happen in case there is more heard speech
            await asyncio.sleep(0)

            logger.debug("Heard speech - now responding")
            while not self.events_log:
                response_tokens_so_far, next_response_token, say_response_so_far, is_response_finished = self._generate_response_segment(
                    state, response_tokens_so_far, next_response_token, llm_cache)
                state = "RESPONDING"
                
                if say_response_so_far:
                    response_so_far = self.tokenizer.decode(response_tokens_so_far)
                    response_tokens_so_far = []
                    logger.info(f"Speaking response: {response_so_far}")
                    await self.pubsub.publish(EventType.RESPONSE_TEXT_GENERATED, response_so_far)
                
                if is_response_finished:
                    # Close out response in LLM cache and start new user message segment
                    _, llm_cache = self._model_call(mx.array([self.tokenizer.eos_token_id]), llm_cache)
                    for t in mx.array(self.tokenizer.encode("[INST]", add_special_tokens=False)):
                        _, llm_cache = self._model_call(t, llm_cache)
                    mx.eval(llm_cache)

                    state = "WAITING"
                    logger.debug(f"Response finished - switching to {state} state")
                    break

                # Allow potenital interuption to happen in case there is more heard speech
                await asyncio.sleep(0)

            if self.events_log:
                logger.debug("Interruption detected, handling")
            else:
                logger.debug("Finished response, waiting for next event")
