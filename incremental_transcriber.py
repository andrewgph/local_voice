import asyncio
from asyncio import Queue
import numpy as np
import time
import mlx.core as mx
from webrtcvad import Vad
import noisereduce as nr
import os

from events import EventType

import whisper_mlx.audio as audio
from whisper_mlx.whisper_mlx import sinusoids
from whisper_mlx.tokenizer import get_tokenizer

import logging

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

# Window size is 160 = 10 ms at 16kHz * 2 bytes per sample
AUDIO_BYTES_WINDOW_SIZE = 160 * 2
# Check last 100ms for VAD to identify pauses
AUDIO_BYTES_VAD_CHECK_SIZE = AUDIO_BYTES_WINDOW_SIZE * 10
# Min audio bytes window to transcribe
# Assume anything less than this doesn't contain speech
# Set to around 500ms
AUDIO_BYTES_MIN_TRANSCRIBE_SIZE = AUDIO_BYTES_WINDOW_SIZE * 50

# Model can take up to 3000 mel frames (30s)
MAX_MEL_FRAMES = 3000
# Append ~100ms second of empty frames to help detect the end of speech
MEL_EMPTY_BUFFER_FRAMES = 10
# Min number of mel frames for model
MIN_MEL_FRAMES = 500


# TODO: assumes single channel audio
def audio_bytes_to_np_array(bytes_data):
    arr = np.frombuffer(bytes_data, dtype='<i2')
    arr = arr.astype('float32') / 32768.0
    return arr


class IncrementalTranscriber:

    def __init__(self, pubsub, whisper_mlx_model, log_dir):
        self.log_dir = log_dir
        self.whisper_mlx_model = whisper_mlx_model
        self.tokenizer = get_tokenizer(
            multilingual=whisper_mlx_model.is_multilingual,
            num_languages=whisper_mlx_model.num_languages,
            language="en",
            task="transcribe",
        )

        self.audio_bytes_vad_buffer = b''
        self.audio_bytes_buffer = b''
        self.vad = Vad()
        self.vad.set_mode(2)

        self.event_queue = Queue()
        self.is_running = False
        self.transcribe_task = None

        # Cached audio prefix to improve short audio transcription
        self.audio_prefix = {
            "result_logprob": 0,
            "tokens": [],
            "np_arr": np.array([]),
        }

        self.pubsub = pubsub
        self.pubsub.subscribe(EventType.HEARD_AUDIO, self.handle_heard_audio, priority=0)

    async def handle_heard_audio(self, audio_bytes):
        logger.debug(f"Handling heard audio {len(audio_bytes)} bytes")
        await self.event_queue.put(audio_bytes)
        if not self.is_running:
            logger.debug("Starting transcriber")
            self.is_running = True
            self.transcribe_task = asyncio.create_task(self.transcribe())
        else:
            logger.debug("Transcriber already running")

    async def transcribe(self):
        while self.is_running:
            audio_bytes = await self.event_queue.get()

            logger.debug(f"Transcribing {len(audio_bytes)} bytes")

            if len(audio_bytes) == 0:
                logger.debug("No audio to transcribe")
                continue

            # Fill up the VAD check buffer
            self.audio_bytes_vad_buffer += audio_bytes
            if len(self.audio_bytes_vad_buffer) < AUDIO_BYTES_VAD_CHECK_SIZE:
                logger.debug("Waiting for VAD check buffer to fill up")
                continue

            logger.debug(f"Audio bytes VAD check buffer: {len(self.audio_bytes_vad_buffer)} bytes")

            # Check through the VAD check buffer in small windows
            # Only keep windows which contain speech
            contains_speech = False
            for i in range(0, len(self.audio_bytes_vad_buffer), AUDIO_BYTES_VAD_CHECK_SIZE):
                vad_check_bytes = self.audio_bytes_vad_buffer[i:i+AUDIO_BYTES_VAD_CHECK_SIZE]
                logger.debug(f"VAD checking window of size {len(vad_check_bytes)}")
                current_contains_speech = self._vad_check(vad_check_bytes)
                if current_contains_speech:
                    logger.debug("VAD check found speech - adding to buffer")
                    self.audio_bytes_buffer += vad_check_bytes
                if len(vad_check_bytes) < AUDIO_BYTES_VAD_CHECK_SIZE:
                    # In this case include the last window for contains speech
                    contains_speech = contains_speech or current_contains_speech
                else:
                    contains_speech = current_contains_speech

            # Reset VAD check buffer
            self.audio_bytes_vad_buffer = b''

            logger.debug(f"Audio bytes buffer with speech: {len(self.audio_bytes_buffer)} bytes")

            # If last window doesn't contain speech, try to transcribe
            if not contains_speech:
                logger.debug("No recent speech detected by VAD")

                if len(self.audio_bytes_buffer) == 0:
                    logger.debug("No audio bytes with speech to transcribe")
                    continue
                if len(self.audio_bytes_buffer) < AUDIO_BYTES_MIN_TRANSCRIBE_SIZE:
                    logger.debug(f"Audio bytes buffer too small to transcribe: {len(self.audio_bytes_buffer)} bytes")
                    continue

                audio_arr = audio_bytes_to_np_array(self.audio_bytes_buffer)
                logger.debug(f"audio arr shape: {audio_arr.shape}")
                self.audio_bytes_buffer = b''

                text = self._transcribe_arr(audio_arr)
                if text.strip():
                    await self.pubsub.publish(EventType.HEARD_SPEECH, text.strip())
                    self._record_result(audio_arr, text)
                
                await self.pubsub.publish(EventType.HEARD_PAUSE, None)
            else:
                # TODO: VAD check might be too sensitive and cause accidental interuptions
                logger.debug("VAD check found recent speech")
                await self.pubsub.publish(EventType.HEARD_SPEAKING, None)

    def _transcribe_arr(self, audio_arr):
        start_time = time.time()

        audio_arr = nr.reduce_noise(y=audio_arr, sr=SAMPLING_RATE)
        prefixed_audio_arr = np.concatenate([self.audio_prefix["np_arr"], audio_arr])
        mel = audio.log_mel_spectrogram(prefixed_audio_arr, self.whisper_mlx_model.dims.n_mels)
        mel = mel.reshape(1, *mel.shape)

        num_additional_frames = max(
            MEL_EMPTY_BUFFER_FRAMES + (1 if mel.shape[1] % 2 != 0 else 0),
            MIN_MEL_FRAMES - mel.shape[1])
        mel = mx.concatenate([
            mel,
            mx.zeros((1, num_additional_frames, self.whisper_mlx_model.dims.n_mels), dtype=mx.float32)
        ], axis=1)

        # Drop older audio if we don't fit into max number of mel frames for model
        # TODO: This is assuming that user wont speak continuously for >= 30 seconds
        if mel.shape[1] > MAX_MEL_FRAMES:
            mel = mel[:, -MAX_MEL_FRAMES:, :]

        logger.debug(f"Resizing positional encoder for mel shape: {mel.shape}")
        self.whisper_mlx_model.encoder._positional_embedding = sinusoids(
            mel.shape[1] // 2, self.whisper_mlx_model.dims.n_audio_state).astype(mx.float16)

        # TODO: crude heuristic of 8 tokens per second in original audio array as an upper bound
        max_tokens = max(int(len(audio_arr) / SAMPLING_RATE * 8), 1)
        logger.debug(f"Max tokens: {max_tokens}")

        decoded_tokens = mx.array([list(self.tokenizer.sot_sequence_including_notimestamps) + self.audio_prefix["tokens"]], dtype=mx.int32)
        result_tokens = []
        kv_cache = None
        next_token = None
        result_logprob = 0

        audio_features = self.whisper_mlx_model.encoder(mel)

        for i in range (max_tokens):
            if i == 0:
                logits, kv_cache, _ = self.whisper_mlx_model.decoder(decoded_tokens, audio_features, kv_cache)
            else:
                logits, kv_cache, _ = self.whisper_mlx_model.decoder(next_token.reshape(1, -1), audio_features, kv_cache)
            next_token = mx.argmax(logits[:, -1], axis=-1)
            next_token_value = next_token.item()
            result_logprob += logits[:, -1, next_token_value] - mx.logsumexp(logits[:, -1], axis=-1)

            if i == 0 and next_token_value == self.tokenizer.no_speech:
                logger.debug("whisper no_speech token generated, skipping transcription")
                return ""

            if next_token_value == self.tokenizer.eot:
                break
            result_tokens.append(next_token_value)
            decoded_tokens = mx.concatenate([decoded_tokens, next_token.reshape(1, -1)], axis=-1)

        # Cache a high confidence transcription for future use to improve short audio transcription
        # TODO: using a min tokens length threshold to avoid a high confidence segment that's too short
        result_prob = mx.exp(result_logprob).item()
        result_logprob = result_logprob.item()
        if result_prob > 0.5 and len(result_tokens) >= 5 and (self.audio_prefix["result_logprob"] == 0 or self.audio_prefix["result_logprob"] < result_logprob):
            self.audio_prefix["result_logprob"] = result_logprob
            self.audio_prefix["tokens"] = result_tokens
            self.audio_prefix["np_arr"] = audio_arr
            logger.info(f"New audio prefix: logprob={self.audio_prefix['result_logprob']} tokens={self.audio_prefix['tokens']}")

        end_time = time.time()
        whisper_time_ms = int(1000 * (end_time - start_time))

        text = self.tokenizer.decode(result_tokens) if result_tokens else ""
        logger.info(f"whisper: {whisper_time_ms}ms : {text}")

        return text

    def _vad_check(self, audio_bytes):
        window_count = 0
        speech_count = 0
        for i in range(0, len(audio_bytes), AUDIO_BYTES_WINDOW_SIZE):
            window_count += 1
            if self.vad.is_speech(audio_bytes[i:i+AUDIO_BYTES_WINDOW_SIZE], SAMPLING_RATE):
                speech_count += 1
        logger.debug(f"VAD found speech in {speech_count} / {window_count} windows")
        return speech_count > 0
    
    def _record_result(self, audio_arr, text):
        timestamp_ms = int(time.time() * 1000)
        
        # Save audio as wav file
        os.makedirs(f"{self.log_dir}/wav_files", exist_ok=True)
        audio_filename = f"{self.log_dir}/wav_files/audio_{timestamp_ms}.wav"
        from scipy.io import wavfile
        wavfile.write(audio_filename, SAMPLING_RATE, audio_arr)
        
        # Save text result to txt file  
        os.makedirs(f"{self.log_dir}/transcript_files", exist_ok=True)
        text_filename = f"{self.log_dir}/transcript_files/transcript_{timestamp_ms}.txt"
        with open(text_filename, "w") as f:
            f.write(text)

