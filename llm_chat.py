import asyncio
from asyncio import Queue
import time
import numpy as np
import mlx.core as mx
from enum import Enum, auto

from events import EventType

from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)

class AgentState(Enum):
    WAITING = auto()
    LISTENING = auto()
    RESPONDING = auto()

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

        self.event_queue = Queue()
        self.is_running = False
        self.process_events_task = None
        self.response_task = None

        self.state = AgentState.WAITING
        self.llm_cache = self._init_llm_cache()
        self.is_heard_speaking = False
        self.pubsub.subscribe(EventType.HEARD_SPEECH, lambda data: self.handle_event(EventType.HEARD_SPEECH, data))
        self.pubsub.subscribe(EventType.HEARD_SPEAKING, lambda data: self.handle_event(EventType.HEARD_SPEAKING, data))
        self.pubsub.subscribe(EventType.HEARD_PAUSE, lambda data: self.handle_event(EventType.HEARD_PAUSE, data))

    async def handle_event(self, event_type, data):
        logger.debug(f"Handling event {event_type} with data: {data}")
        await self.event_queue.put((event_type, data))
        if not self.is_running:
            logger.debug("Starting process_events task")
            self.is_running = True
            self.process_events_task = asyncio.create_task(self.process_events())
        else:
            logger.debug("process_events already running")

    async def process_events(self):
        while self.is_running:
            event = await self.event_queue.get()
            logger.debug(f"Processing event: {event}, event type: {event[0]}, data: {event[1]}")
            
            try:
                if event[0] == EventType.HEARD_SPEECH:
                    if event[1].strip() == "":
                        continue

                    # Cancel any existing response task
                    if self.response_task and not self.response_task.done():
                        logger.debug("Heard new speech while responding, cancelling response task")
                        self.response_task.cancel()
                        try:
                            # Wait for the task to be cancelled.
                            # This ensures any cleanup or finalization can occur within the task.
                            await self.response_task
                        except asyncio.CancelledError:
                            logger.debug("Response generation task was cancelled.")
                        finally:
                            self.response_task = None  # Reset the task to None after cancellation

                    self._update_heard_speech(event[1])

                elif event[0] == EventType.HEARD_SPEAKING:
                    self.is_heard_speaking = True
                
                elif event[0] == EventType.HEARD_PAUSE:
                    if self.state == AgentState.WAITING:
                        logger.debug("Paused while waiting for user input, ignoring")
                        continue

                    self.is_heard_speaking = False
                    if self.response_task is None or self.response_task.done():
                        logger.debug("Creating new response task")
                        self.response_task = asyncio.create_task(self._generate_response())
                    else:
                        logger.debug("Response task already running")
            except Exception as e:
                logger.error(f"Error processing event {event[0]}: {str(e)}")

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

    def _update_heard_speech(self, transcript):
        logger.debug(f"LLM handling heard speech: {transcript}")

        if self.state == AgentState.RESPONDING:
            logger.debug("Interruption detected, closing response segment")
            # Close out response attempt
            _, self.llm_cache = self._model_call(mx.array([self.tokenizer.eos_token_id]), self.llm_cache)
        
        if self.state != AgentState.LISTENING:
            logger.debug("Starting new user message segment")
            # Start new user message segment
            for t in mx.array(self.tokenizer.encode("[INST]", add_special_tokens=False)):
                _, self.llm_cache = self._model_call(t, self.llm_cache)
            mx.eval(self.llm_cache)
        
        logger.debug(f"In state {self.state}, switching to LISTENING")
        self.state = AgentState.LISTENING

        logger.debug(f"Updating LLM cache with: {transcript}")
        for t in mx.array(self.tokenizer.encode(transcript, add_special_tokens=False)):
            _, self.llm_cache = self._model_call(t, self.llm_cache)
        mx.eval(self.llm_cache)
        logger.debug("Updated LLM cache")

    async def _generate_response(self):
        logger.debug("Generating response")

        next_response_token = None
        response_tokens_so_far = []

        while True:
            # Pause to let other tasks process
            await asyncio.sleep(0)

            if self.is_heard_speaking:
                # Wait to see if new speech is heard, response will be cancelled in that case
                # TODO: this could be based upon a pause flag
                logger.debug("Heard speaking while generating response - waiting for pause")
                await asyncio.sleep(0)
                logger.debug("Resuming response generation after pause for user speaking")
                continue

            if self.state != AgentState.RESPONDING:
                logger.debug(f"In state {self.state}, switching to RESPONDING, starting new response segment")
                response_tokens_so_far = []
                for t in mx.array(self.tokenizer.encode("[/INST]", add_special_tokens=False)):
                    logits, self.llm_cache = self._model_call(t, self.llm_cache)
                next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
                logger.debug("Added response tokens")

            logger.debug(f"In state {self.state}, switching to RESPONDING")
            self.state = AgentState.RESPONDING

            assert next_response_token is not None, "expected next_response_token to be initialized in responding state"
        
            say_response_so_far = False
            is_response_finished = False
            found_punctuation = False
            for _ in range(self.config.num_tokens_response_pause):
                if next_response_token.item() == self.tokenizer.eos_token_id:
                    is_response_finished = True
                    logger.debug("Found end of response, breaking response generation")
                    break

                response_tokens_so_far.append(next_response_token.item())
                decoded_next_token = self.tokenizer.decode([next_response_token.item()])

                # also check for any punctuation, as that indicates end of a sentence, and we
                # want to say each sentence asap
                if any(punct in decoded_next_token for punct in ',.!?'):
                    found_punctuation = True
                    logger.debug("Found speech break punctuation, breaking response generation")
                    break

                logits, self.llm_cache = self._model_call(next_response_token, self.llm_cache)
                next_response_token = mx.argmax(logits.squeeze(1), axis=-1)

            if is_response_finished:
                logger.debug("Finished response")
                say_response_so_far = True

            elif found_punctuation:
                logger.debug("Found punctuation indicating end of a sentence")
                say_response_so_far = True

                # Generate a new next token to avoid immediately breaking on next iteration
                logits, self.llm_cache = self._model_call(next_response_token, self.llm_cache)
                next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
                mx.eval(next_response_token)

            else:
                # else we are just pausing response to check for new audio,
                # and will keep generating if there are no interuptions
                logger.debug(f"Pausing to check for interuptions with partial response: {self.tokenizer.decode(response_tokens_so_far)}")
                mx.eval(next_response_token)
            
            if say_response_so_far:
                response_so_far = self.tokenizer.decode(response_tokens_so_far)
                response_tokens_so_far = []
                logger.info(f"Speaking response: {response_so_far}")
                await self.pubsub.publish(EventType.RESPONSE_TEXT_GENERATED, response_so_far)
            
            if is_response_finished:
                # Close out response in LLM cache and start new user message segment
                _, self.llm_cache = self._model_call(mx.array([self.tokenizer.eos_token_id]), self.llm_cache)
                for t in mx.array(self.tokenizer.encode("[INST]", add_special_tokens=False)):
                    _, self.llm_cache = self._model_call(t, self.llm_cache)
                mx.eval(self.llm_cache)

                logger.debug(f"In state {self.state}, switching to WAITING")
                self.state = AgentState.WAITING
                return
