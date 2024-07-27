import asyncio
from asyncio import Queue
import numpy as np
import time
import mlx.core as mx
from webrtcvad import Vad
import os
import json
from dataclasses import dataclass

from events import EventType

import whisper_mlx.audio as audio
from whisper_mlx.whisper_mlx import sinusoids
from whisper_mlx.tokenizer import get_tokenizer
from audio_io import SAMPLE_RATE

import logging

logger = logging.getLogger(__name__)


# Min audio bytes window to transcribe
# Assume anything less than this doesn't contain speech
# Set to 500ms: 16 samples every 1ms * 2 bytes per sample * 500ms
AUDIO_BYTES_MIN_TRANSCRIBE_SIZE = 16 * 2 * 500

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


@dataclass
class TranscriptionResult:
    text: str
    prob: float
    logprob: float
    tokens: list[int]
    audio_arr: np.ndarray


def transcribe(audio_arr, whisper_mlx_model, tokenizer, audio_prefix=None):
    logger.debug(f"audio arr shape: {audio_arr.shape}")

    if audio_prefix is not None:
        prefixed_audio_arr = np.concatenate([audio_prefix["np_arr"], audio_arr])
    else:
        prefixed_audio_arr = audio_arr

    mel = audio.log_mel_spectrogram(prefixed_audio_arr, whisper_mlx_model.dims.n_mels)
    mel = mel.reshape(1, *mel.shape)
    num_additional_frames = max(
        MEL_EMPTY_BUFFER_FRAMES + (1 if mel.shape[1] % 2 != 0 else 0),
        MIN_MEL_FRAMES - mel.shape[1])
    mel = mx.concatenate([
        mel,
        mx.zeros((1, num_additional_frames, whisper_mlx_model.dims.n_mels), dtype=mx.float32)
    ], axis=1)

    if mel.shape[1] > MAX_MEL_FRAMES:
        mel = mel[:, -MAX_MEL_FRAMES:, :]

    logger.debug(f"Resizing positional encoder for mel shape: {mel.shape}")
    whisper_mlx_model.encoder._positional_embedding = sinusoids(
        mel.shape[1] // 2, whisper_mlx_model.dims.n_audio_state).astype(mx.float16)

    max_tokens = max(int(len(audio_arr) / SAMPLE_RATE * 8), 1)
    logger.debug(f"Max tokens: {max_tokens}")

    if audio_prefix is not None:
        decoded_tokens = mx.array([list(tokenizer.sot_sequence_including_notimestamps) + audio_prefix["tokens"]], dtype=mx.int32)
    else:
        decoded_tokens = mx.array([list(tokenizer.sot_sequence_including_notimestamps)], dtype=mx.int32)
    result_tokens = []
    kv_cache = None
    next_token = None
    result_logprob = 0

    audio_features = whisper_mlx_model.encoder(mel)

    for i in range(max_tokens):
        if i == 0:
            logits, kv_cache, _ = whisper_mlx_model.decoder(decoded_tokens, audio_features, kv_cache)
        else:
            logits, kv_cache, _ = whisper_mlx_model.decoder(next_token.reshape(1, -1), audio_features, kv_cache)
        next_token = mx.argmax(logits[:, -1], axis=-1)
        next_token_value = next_token.item()
        result_logprob += logits[:, -1, next_token_value] - mx.logsumexp(logits[:, -1], axis=-1)

        if i == 0 and next_token_value == tokenizer.no_speech:
            logger.debug("whisper no_speech token generated, skipping transcription")
            return ""

        if next_token_value == tokenizer.eot:
            break
        result_tokens.append(next_token_value)
        decoded_tokens = mx.concatenate([decoded_tokens, next_token.reshape(1, -1)], axis=-1)

    result_prob = mx.exp(result_logprob).item()
    result_logprob = result_logprob.item()

    text = tokenizer.decode(result_tokens) if result_tokens else ""

    return TranscriptionResult(
        text=text,
        prob=result_prob,
        logprob=result_logprob,
        tokens=result_tokens,
        audio_arr=audio_arr
    )


class IncrementalTranscriber:

    def __init__(self, pubsub, whisper_mlx_model, log_dir, calibration_dir="calibration"):
        self.log_dir = log_dir
        self.calibration_dir = calibration_dir
        os.makedirs(self.calibration_dir, exist_ok=True)
        self.whisper_mlx_model = whisper_mlx_model
        self.tokenizer = get_tokenizer(
            multilingual=whisper_mlx_model.is_multilingual,
            num_languages=whisper_mlx_model.num_languages,
            language="en",
            task="transcribe",
        )

        self.audio_bytes_buffer = b''

        self.event_queue = Queue()
        self.is_running = False
        self.event_loop_task = None
        self.transcribe_task = None

        # Cached audio prefix to improve short audio transcription
        self.audio_prefix = {
            "result_logprob": 0,
            "tokens": [],
            "np_arr": np.array([]),
        }
        # Load a cached audio prefix if it exists
        self.allow_calibration_update = True
        self._load_calibration_data()

        self.pubsub = pubsub
        self.pubsub.subscribe(EventType.HEARD_SPEAKING, self.handle_heard_speaking, priority=5)
        self.pubsub.subscribe(EventType.HEARD_NO_SPEAKING, self.handle_heard_no_speaking, priority=5)

    def _load_calibration_data(self):
        logger.debug("Attempting to load calibration data")
        file_path = os.path.join(self.calibration_dir, 'audio_prefix.json')
        try:
            with open(file_path, 'r') as file:
                audio_prefix_data = json.load(file)
                self.audio_prefix['result_logprob'] = audio_prefix_data['result_logprob']
                self.audio_prefix['tokens'] = audio_prefix_data['tokens']
                self.audio_prefix['np_arr'] = np.array(audio_prefix_data['np_arr'])
                # Don't update existing calibration data that was created before this session
                self.allow_calibration_update = False
        except FileNotFoundError:
            logger.warning(f"Calibration file not found at {file_path}. Using default calibration data.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from the calibration file at {file_path}.")
        except Exception as e:
            logger.error(f"An error occurred while loading calibration data: {str(e)}")

    def _save_calibration_data(self):
        logger.debug("Attempting to save calibration data")
        file_path = os.path.join(self.calibration_dir, 'audio_prefix.json')
        with open(file_path, 'w') as file:
            # Convert numpy array to list for JSON serialization
            audio_prefix_copy = self.audio_prefix.copy()
            audio_prefix_copy['np_arr'] = self.audio_prefix['np_arr'].tolist()
            json.dump(audio_prefix_copy, file)

    def _start_transcriber(self):
        logger.debug("Starting transcriber")
        self.is_running = True
        self.event_loop_task = asyncio.create_task(self.event_loop())

    async def handle_heard_speaking(self, audio_bytes):
        logger.debug(f"Handling heard speaking {len(audio_bytes)} bytes")
        await self.event_queue.put(audio_bytes)
        if not self.is_running:
            self._start_transcriber()
        else:
            logger.debug("Transcriber already running")
    
    async def handle_heard_no_speaking(self, data):
        logger.debug("Handling heard no speaking")
        await self.event_queue.put(None)
        if not self.is_running:
            self._start_transcriber()
        else:
            logger.debug("Transcriber already running")

    async def event_loop(self):
        while self.is_running:
            audio_bytes = await self.event_queue.get()

            if audio_bytes is not None:
                # Cancel any existing transcribe task if we have new speech audio before it completes
                if self.transcribe_task is not None:
                    logger.debug("Cancelling existing transcribe task")
                    self.transcribe_task.cancel()
                    self.transcribe_task = None

                self.audio_bytes_buffer += audio_bytes
                logger.debug(f"Adding {len(audio_bytes)} bytes to buffer, new buffer size {len(self.audio_bytes_buffer)} bytes")
                continue

            # Heard no speaking, checking if there is audio to transcribe
            logger.debug(f"Transcribing audio bytes buffer with speech: {len(self.audio_bytes_buffer)} bytes")

            if len(self.audio_bytes_buffer) == 0:
                logger.debug("No audio bytes with speech to transcribe")
                continue
            if len(self.audio_bytes_buffer) < AUDIO_BYTES_MIN_TRANSCRIBE_SIZE:
                logger.debug(f"Audio bytes buffer too small to transcribe: {len(self.audio_bytes_buffer)} bytes")
                continue

            # Start a new transcription task only if it's not already running
            if self.transcribe_task is None:
                logger.debug("Starting new transcribe task")
                self.transcribe_task = asyncio.create_task(self._transcribe())

    async def _transcribe(self):
        # TODO: Move transcription to a separate thread so the MLX operations don't block other asyncio tasks
        # This doesn't currently work as VAD is too sensitive, causes continual interruptions to transcription task
        #text = await asyncio.to_thread(self._transcribe_arr)
        text = self._transcribe_arr()

        self._record_result(audio_bytes_to_np_array(self.audio_bytes_buffer), text)
        if text.strip():
            await self.pubsub.publish(EventType.HEARD_SPEECH, text.strip())
        self.audio_bytes_buffer = b''

    def _transcribe_arr(self):
        start_time = time.time()

        audio_arr = audio_bytes_to_np_array(self.audio_bytes_buffer)

        transcription_result = transcribe(
            audio_arr,
            self.whisper_mlx_model,
            self.tokenizer,
            self.audio_prefix
        )

        # Include a min token length as otherwise there is a bias towards shorter transcriptions which have higher probability
        if self.allow_calibration_update and \
            transcription_result.prob > 0.25 and \
            len(transcription_result.tokens) >= 5 and \
            (self.audio_prefix["result_logprob"] == 0 or self.audio_prefix["result_logprob"] < transcription_result.logprob):
            self.audio_prefix["result_logprob"] = transcription_result.logprob
            self.audio_prefix["tokens"] = transcription_result.tokens
            self.audio_prefix["np_arr"] = transcription_result.audio_arr
            logger.info(f"New audio prefix: logprob={self.audio_prefix['result_logprob']} tokens={self.audio_prefix['tokens']}")
            self._save_calibration_data()

        end_time = time.time()
        whisper_time_ms = int(1000 * (end_time - start_time))

        logger.info(f"whisper: {whisper_time_ms}ms : {transcription_result.text} ({transcription_result.prob:.2f})")

        return transcription_result.text

    def _record_result(self, audio_arr, text):
        timestamp_ms = int(time.time() * 1000)
        
        os.makedirs(f"{self.log_dir}/wav_files", exist_ok=True)
        audio_filename = f"{self.log_dir}/wav_files/audio_{timestamp_ms}.wav"
        from scipy.io import wavfile
        wavfile.write(audio_filename, SAMPLE_RATE, audio_arr)
        
        os.makedirs(f"{self.log_dir}/transcript_files", exist_ok=True)
        text_filename = f"{self.log_dir}/transcript_files/transcript_{timestamp_ms}.txt"
        with open(text_filename, "w") as f:
            f.write(text)
