from enum import Enum, auto
import os
import time

import mlx.core as mx


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

# https://github.com/ml-explore/mlx-examples/blob/85dc76f6e0f2cf3ee3d84c211868a6856e163f3f/llms/mlx_lm/models/base.py#L9
class KVCache:

    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


def load_chat_model(model, tokenizer, logs_dir):
    model_class_name = f"{model.__class__.__module__}.{model.__class__.__name__}"
    if model_class_name == "mlx_lm.models.llama.Model":
        return LlamaChatModel(model, tokenizer, logs_dir)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")


class ChatModel:

    def initialize(self, system_instructions, initial_messages):
        pass
        
    def add_user_message_segment(self, segment):
        pass

    def prob_end_of_user_message(self):
        pass
    
    def generate_response_segment(self, num_tokens):
        pass


class LlamaChatModel(ChatModel):

    def __init__(self, llama_model, llama_tokenizer, logs_dir):
        os.makedirs(f"{logs_dir}/token_logs", exist_ok=True)
        timestamp_ms = int(time.time() * 1000)
        self.token_log_file = f"{logs_dir}/token_logs/chat_tokens_{timestamp_ms}.txt"

        self.model = llama_model
        self.tokenizer = llama_tokenizer
        self._initialize_cache()
        self.state = ChatModelState.WAITING
        self.last_logits = None

    def _initialize_cache(self):
        kv_heads = (
            [self.model.n_kv_heads] * len(self.model.layers)
            if isinstance(self.model.n_kv_heads, int)
            else self.model.n_kv_heads
        )
        self.kv_cache = [KVCache(self.model.head_dim, n) for n in kv_heads]

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

        self._log_tokens(input_ids)

        mx.eval(self.model(mx.array(input_ids)[None], self.kv_cache))
    
    def add_user_message_segment(self, text):
        if self.state == ChatModelState.ASSISTANT_TURN:
            logger.debug("Interruption detected, closing assistant segment")
            # Close out response attempt
            self._model_call(mx.array(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)), self.kv_cache)
        
        if self.state != ChatModelState.USER_TURN:
            logger.debug("Starting new user message segment")
            # Start new user message segment
            for t in mx.array(self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n", add_special_tokens=False)):
                self._model_call(t, self.kv_cache)
        
        logger.debug(f"In state {self.state}, switching to USER_TURN")
        self.state = ChatModelState.USER_TURN

        logger.debug(f"Updating LLM cache with: {text}")
        for t in mx.array(self.tokenizer.encode(text, add_special_tokens=False)):
            self.last_logits = self._model_call(t, self.kv_cache)
        mx.eval(self.last_logits)
        logger.debug("Updated LLM cache")

    def prob_end_of_user_message(self):
        probs = self.last_logits[0, -1, self.tokenizer.eos_token_id] - mx.logsumexp(self.last_logits[0, -1], axis=-1)
        return mx.exp(probs).item()

    def generate_response_segment(self, response_tokens_so_far, next_response_token, num_tokens):
        is_response_finished = False
        say_response_so_far = False
        found_punctuation = False

        if self.state != ChatModelState.ASSISTANT_TURN:
            logger.debug(f"In state {self.state}, switching to ASSISTANT_TURN, starting new response segment")
            response_tokens_so_far = []
            for t in mx.array(self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)):
                logits = self._model_call(t, self.kv_cache)
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

            logits = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)

        if found_punctuation:
            logger.debug("Found punctuation indicating end of a sentence")
            say_response_so_far = True

            # Generate a new next token to avoid immediately breaking on next iteration
            logits = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            mx.eval(next_response_token)
        
        if is_response_finished:
            logger.debug("Finished response - ending assistant section")
            # Close out response in LLM cache
            self._model_call(mx.array([self.tokenizer.eos_token_id]), self.kv_cache)
            for t in mx.array(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)):
                self._model_call(t, self.kv_cache)

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
        self._log_tokens([token.item()])
        return self.model(token.reshape(1, 1), llm_cache)

    def _log_tokens(self, tokens):
        with open(self.token_log_file, "a") as file:
            for token in tokens:
                file.write(str(token) + "\n")
                file.flush()