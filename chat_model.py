from enum import Enum, auto

import mlx.core as mx
from mlx_lm.utils import load as load_llm_model

import logging

logger = logging.getLogger(__name__)

class ChatModelState(Enum):
    WAITING = auto()
    USER_TURN = auto()
    ASSISTANT_TURN = auto()


from dataclasses import dataclass

@dataclass
class ResponseSegmentResult:
    text: str
    tokens: list
    next_response_token: int
    say_segment: bool
    is_complete: bool


def load_chat_model(model_path):
    model, tokenizer = load_llm_model(model_path)
    model_class_name = f"{model.__class__.__module__}.{model.__class__.__name__}"
    if model_class_name == "mlx_lm.models.llama.Model":
        return LlamaChatModel(model, tokenizer)
    elif model_class_name in ["mlx_lm.models.mistral.Model", "mlx_lm.models.mixtral.Model"]:
        return MistralChatModel(model, tokenizer)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")


class ChatModel:

    def initialize(self, system_instructions, initial_messages):
        pass
        
    def add_user_message_segment(self, segment):
        pass
    
    def generate_response_segment(self, num_tokens):
        pass


class LlamaChatModel(ChatModel):

    def __init__(self, llama_model, llama_tokenizer):
        self.model = llama_model
        self.tokenizer = llama_tokenizer
        self.kv_cache = None
        self.state = ChatModelState.WAITING

    def initialize(self, system_instructions, initial_messages):
        messages = [
            {"role": "system", "content": system_instructions},
        ]

        for heard, response in initial_messages:
            messages.append({"role": "user", "content": heard})
            messages.append({"role": "assistant", "content": response})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        _, self.kv_cache = self.model(mx.array(input_ids)[None])
        mx.eval(self.kv_cache)
    
    def add_user_message_segment(self, text):
        if self.state == ChatModelState.ASSISTANT_TURN:
            logger.debug("Interruption detected, closing assistant segment")
            # Close out response attempt
            _, self.kv_cache = self._model_call(mx.array(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)), self.kv_cache)
        
        if self.state != ChatModelState.USER_TURN:
            logger.debug("Starting new user message segment")
            # Start new user message segment
            for t in mx.array(self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n", add_special_tokens=False)):
                _, self.kv_cache = self._model_call(t, self.kv_cache)
            mx.eval(self.kv_cache)
        
        logger.debug(f"In state {self.state}, switching to USER_TURN")
        self.state = ChatModelState.USER_TURN

        logger.debug(f"Updating LLM cache with: {text}")
        for t in mx.array(self.tokenizer.encode(text, add_special_tokens=False)):
            _, self.kv_cache = self._model_call(t, self.kv_cache)
        mx.eval(self.kv_cache)
        logger.debug("Updated LLM cache")

    def generate_response_segment(self, response_tokens_so_far, next_response_token, num_tokens):
        is_response_finished = False
        say_response_so_far = False
        found_punctuation = False

        if self.state != ChatModelState.ASSISTANT_TURN:
            logger.debug(f"In state {self.state}, switching to ASSISTANT_TURN, starting new response segment")
            response_tokens_so_far = []
            for t in mx.array(self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)):
                logits, self.kv_cache = self._model_call(t, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            logger.debug("Added response tokens")

        logger.debug(f"In state {self.state}, switching to ASSISTANT_TURN")
        self.state = ChatModelState.ASSISTANT_TURN

        for _ in range(num_tokens):
            if next_response_token.item() == self.tokenizer.eos_token_id:
                is_response_finished = True
                logger.debug("Found end of response token, breaking response generation")
                break

            response_tokens_so_far.append(next_response_token.item())
            decoded_next_token = self.tokenizer.decode([next_response_token.item()])

            # also check for any punctuation, as that indicates end of a sentence, and we
            # want to say each sentence asap
            if any(punct in decoded_next_token for punct in ',.!?'):
                found_punctuation = True
                logger.debug("Found speech break punctuation, breaking response generation")
                break

            logits, self.kv_cache = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)

        if found_punctuation:
            logger.debug("Found punctuation indicating end of a sentence")
            say_response_so_far = True

            # Generate a new next token to avoid immediately breaking on next iteration
            logits, self.kv_cache = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            mx.eval(next_response_token)
        
        if is_response_finished:
            logger.debug("Finished response - ending assistant section")
            # Close out response in LLM cache
            _, self.kv_cache = self._model_call(mx.array([self.tokenizer.eos_token_id]), self.kv_cache)
            for t in mx.array(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)):
                _, self.kv_cache = self._model_call(t, self.kv_cache)
            mx.eval(self.kv_cache)

            say_response_so_far = True

            logger.debug(f"In state {self.state}, switching to WAITING")
            self.state = ChatModelState.WAITING

        return ResponseSegmentResult(
            text=self.tokenizer.decode(response_tokens_so_far),
            tokens=response_tokens_so_far,
            next_response_token=next_response_token,
            say_segment=say_response_so_far,
            is_complete=is_response_finished
        )

    def _model_call(self, token, llm_cache):
        # token_str = str(token.item())
        # with open("tokens_log.txt", "a") as file:
        #     file.write(token_str + "\n")
        #     file.flush()
        return self.model(token.reshape(1, 1), llm_cache)


class MistralChatModel(ChatModel):

    def __init__(self, mistral_model, mistral_tokenizer):
        self.model = mistral_model
        self.tokenizer = mistral_tokenizer
        self.kv_cache = None
        self.state = ChatModelState.WAITING
    
    def initialize(self, system_instructions, initial_messages):
        tokens = [self.tokenizer.bos_token_id]
        tokens += self.tokenizer.encode("[INST]", add_special_tokens=False)
        tokens += self.tokenizer.encode(system_instructions, add_special_tokens=False)
        tokens += self.tokenizer.encode(initial_messages[0][0], add_special_tokens=False)
        tokens += self.tokenizer.encode("[/INST]", add_special_tokens=False)
        tokens += self.tokenizer.encode(initial_messages[0][1], add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id)

        for heard, response in initial_messages[1:]:
            tokens += self.tokenizer.encode("[INST]", add_special_tokens=False)
            tokens += self.tokenizer.encode(heard, add_special_tokens=False)
            tokens += self.tokenizer.encode("[/INST]", add_special_tokens=False)
            tokens += self.tokenizer.encode(response, add_special_tokens=False)
            tokens.append(self.tokenizer.eos_token_id)

        # Start an initial user message segment
        tokens += self.tokenizer.encode("[INST]", add_special_tokens=False)

        _, self.kv_cache = self.model(mx.array(tokens)[None])
        mx.eval(self.kv_cache)
        
    def add_user_message_segment(self, text):        
        if self.state == ChatModelState.ASSISTANT_TURN:
            logger.debug("Interruption detected, closing assistant segment")
            # Close out response attempt
            _, self.kv_cache = self._model_call(mx.array([self.tokenizer.eos_token_id]), self.kv_cache)
        
        if self.state != ChatModelState.USER_TURN:
            logger.debug("Starting new user message segment")
            # Start new user message segment
            for t in mx.array(self.tokenizer.encode("[INST]", add_special_tokens=False)):
                _, self.kv_cache = self._model_call(t, self.kv_cache)
            mx.eval(self.kv_cache)
        
        logger.debug(f"In state {self.state}, switching to USER_TURN")
        self.state = ChatModelState.USER_TURN

        logger.debug(f"Updating LLM cache with: {text}")
        for t in mx.array(self.tokenizer.encode(text, add_special_tokens=False)):
            _, self.kv_cache = self._model_call(t, self.kv_cache)
        mx.eval(self.kv_cache)
        logger.debug("Updated LLM cache")

    def generate_response_segment(self, response_tokens_so_far, next_response_token, num_tokens):
        is_response_finished = False
        say_response_so_far = False
        found_punctuation = False

        if self.state != ChatModelState.ASSISTANT_TURN:
            logger.debug(f"In state {self.state}, switching to ASSISTANT_TURN, starting new response segment")
            response_tokens_so_far = []
            for t in mx.array(self.tokenizer.encode("[/INST]", add_special_tokens=False)):
                logits, self.kv_cache = self._model_call(t, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            logger.debug("Added response tokens")

        logger.debug(f"In state {self.state}, switching to ASSISTANT_TURN")
        self.state = ChatModelState.ASSISTANT_TURN

        for _ in range(num_tokens):
            if next_response_token.item() == self.tokenizer.eos_token_id:
                is_response_finished = True
                logger.debug("Found end of response token, breaking response generation")
                break

            response_tokens_so_far.append(next_response_token.item())
            decoded_next_token = self.tokenizer.decode([next_response_token.item()])

            # also check for any punctuation, as that indicates end of a sentence, and we
            # want to say each sentence asap
            if any(punct in decoded_next_token for punct in ',.!?'):
                found_punctuation = True
                logger.debug("Found speech break punctuation, breaking response generation")
                break

            logits, self.kv_cache = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)

        if found_punctuation:
            logger.debug("Found punctuation indicating end of a sentence")
            say_response_so_far = True

            # Generate a new next token to avoid immediately breaking on next iteration
            logits, self.kv_cache = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            mx.eval(next_response_token)
        
        if is_response_finished:
            logger.debug("Finished response - ending assistant section")
            # Close out response in LLM cache and start new user message segment
            _, self.kv_cache = self._model_call(mx.array([self.tokenizer.eos_token_id]), self.kv_cache)
            for t in mx.array(self.tokenizer.encode("[INST]", add_special_tokens=False)):
                _, self.kv_cache = self._model_call(t, self.kv_cache)
            mx.eval(self.kv_cache)

            say_response_so_far = True

            logger.debug(f"In state {self.state}, switching to WAITING")
            self.state = ChatModelState.WAITING

        return ResponseSegmentResult(
            text=self.tokenizer.decode(response_tokens_so_far),
            tokens=response_tokens_so_far,
            next_response_token=next_response_token,
            say_segment=say_response_so_far,
            is_complete=is_response_finished
        )

    def _model_call(self, token, llm_cache):
        # token_str = str(token.item())
        # with open("tokens_log.txt", "a") as file:
        #     file.write(token_str + "\n")
        #     file.flush()
        return self.model(token.reshape(1, 1), llm_cache)