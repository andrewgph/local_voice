from enum import Enum, auto
import os
import time

import mlx.core as mx
from mlx_lm.models.base import KVCache

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
    """
    Llama 3 instruct chat model.
    
    This code is specific to Llama 3 as it depends upon the Llama 3 token format for chat.
    The tokenizer chat template isn't used as we want to control when the tokens are added
    for turn taking.
    """

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
        start_time = time.time()
        num_tokens_added = 0

        if self.state == ChatModelState.ASSISTANT_TURN:
            logger.debug("Interruption detected, closing assistant segment")
            # Close out response attempt
            tokens = mx.array(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False))
            self._model_call(tokens, self.kv_cache)
            num_tokens_added += len(tokens)
        
        if self.state != ChatModelState.USER_TURN:
            logger.debug("Starting new user message segment")
            # Start new user message segment
            tokens = mx.array(self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n", add_special_tokens=False))
            self._model_call(tokens, self.kv_cache)
            num_tokens_added += len(tokens)
        
        logger.debug(f"In state {self.state}, switching to USER_TURN")
        self.state = ChatModelState.USER_TURN

        logger.debug(f"Updating LLM cache with: {text}")
        tokens = mx.array(self.tokenizer.encode(text, add_special_tokens=False))
        self.last_logits = self._model_call(tokens, self.kv_cache)[:, -1:, :]
        mx.eval(self.last_logits)
        num_tokens_added += len(tokens)
        logger.debug("Updated LLM cache")

        end_time = time.time()
        generation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        tokens_per_second = (num_tokens_added / generation_time) * 1000  # Convert ms to seconds
        logger.info(f"LLM adding user message segment took {generation_time:.2f} ms, added {num_tokens_added} tokens ({tokens_per_second:.2f} tokens/s)")

    def prob_end_of_user_message(self):
        probs = self.last_logits[0, -1, self.tokenizer.eos_token_id] - mx.logsumexp(self.last_logits[0, -1], axis=-1)
        return mx.exp(probs).item()

    def generate_response_segment(self, response_tokens_so_far, next_response_token, num_tokens):
        start_time = time.time()
        num_tokens_added = 0

        is_response_finished = False
        say_response_so_far = False
        found_punctuation = False

        if self.state != ChatModelState.ASSISTANT_TURN:
            logger.debug(f"In state {self.state}, switching to ASSISTANT_TURN, starting new response segment")
            response_tokens_so_far = []
            tokens = mx.array(self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False))
            logits = self._model_call(tokens, self.kv_cache)[:, -1:, :]
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            logger.debug("Added response tokens")
            num_tokens_added += len(tokens)

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
            num_tokens_added += 1

        if found_punctuation:
            logger.debug("Found punctuation indicating end of a sentence")
            say_response_so_far = True

            # Generate a new next token to avoid immediately breaking on next iteration
            logits = self._model_call(next_response_token, self.kv_cache)
            next_response_token = mx.argmax(logits.squeeze(1), axis=-1)
            num_tokens_added += 1
            mx.eval(next_response_token)
        
        if is_response_finished:
            logger.debug("Finished response - ending assistant section")
            # Close out response in LLM cache
            tokens = mx.array(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False))
            mx.eval(self._model_call(tokens, self.kv_cache))
            num_tokens_added += len(tokens)

            say_response_so_far = True

            logger.debug(f"In state {self.state}, switching to WAITING")
            self.state = ChatModelState.WAITING

        end_time = time.time()
        generation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        tokens_per_second = (num_tokens_added / generation_time) * 1000  # Convert ms to seconds
        logger.info(f"LLM response segment generation took {generation_time:.2f} ms, added {num_tokens_added} tokens ({tokens_per_second:.2f} tokens/s)")

        return ResponseSegmentResult(
            text=self.tokenizer.decode(response_tokens_so_far),
            tokens=response_tokens_so_far,
            next_response_token=next_response_token,
            say_segment=say_response_so_far,
            is_complete=is_response_finished
        )

    def _model_call(self, tokens, llm_cache):
        self._log_tokens([token.item() for token in tokens])
        return self.model(tokens[None], llm_cache)

    def _log_tokens(self, tokens):
        with open(self.token_log_file, "a") as file:
            for token in tokens:
                file.write(str(token) + "\n")
                file.flush()