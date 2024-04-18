import argparse
import numpy as np
import time
import math
import mlx.core as mx
from webrtcvad import Vad
import noisereduce as nr
from dataclasses import dataclass

from events import EventType

import whisper_mlx.audio as audio
from whisper_mlx.whisper_mlx import sinusoids
from whisper_mlx.tokenizer import get_tokenizer

import logging

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

@dataclass
class TranscriptionResult:
    text: str
    tokens: mx.array


# TODO: assumes single channel audio
def audio_bytes_to_np_array(bytes_data):
    arr = np.frombuffer(bytes_data, dtype='<i2')
    arr = arr.astype('float32') / 32768.0
    return arr

def record_result(audio_arr, text):
    timestamp_ms = int(time.time() * 1000)
    
    # Save audio as wav file
    audio_filename = f"transcribe_log/audio_{timestamp_ms}.wav"
    from scipy.io import wavfile
    wavfile.write(audio_filename, SAMPLING_RATE, audio_arr)
    
    # Save text result to txt file  
    text_filename = f"transcribe_log/text_{timestamp_ms}.txt"
    with open(text_filename, "w") as f:
        f.write(text)


class IncrementalTranscriber:

    def __init__(self, pubsub, whisper_mlx_model, num_mel_frames=500):
        assert num_mel_frames % 2 == 0, "num_mel_frames must be even"

        self.pubsub = pubsub
        self.pubsub.subscribe(EventType.HEARD_AUDIO, self.handle_heard_audio)

        self.whisper_mlx_model = whisper_mlx_model
        self.tokenizer = get_tokenizer(
            multilingual=whisper_mlx_model.is_multilingual,
            num_languages=whisper_mlx_model.num_languages,
            language="en",
            task="transcribe",
        )
        self.num_mel_frames = num_mel_frames
        self.num_audio_arr_samples = num_mel_frames * 160 # each frame 10ms * 160 samples for 10ms at 16000 hz
        self.num_empty_buffer_frames = 10
        # Shorten the positional embedding for the expected number of frames + empty buffer
        self.whisper_mlx_model.encoder._positional_embedding = sinusoids(
            (self.num_mel_frames + self.num_empty_buffer_frames) // 2, whisper_mlx_model.dims.n_audio_state).astype(mx.float16)

        # Maintain buffer of desired size
        # New audio will be appended to it and old audio will be dropped from the beginning
        self.audio_arr = np.array([])
        self.audio_bytes_buffer = b''
        self.previous_contains_speech = False

        self.vad = Vad()
        self.vad.set_mode(3)

    async def handle_heard_audio(self, audio_bytes):
        await self.transcribe(audio_bytes)

    async def transcribe(self, audio_bytes):
        logger.debug(f"Transcribing {len(audio_bytes)} bytes")

        if len(audio_bytes) == 0:
            logger.debug("No audio to transcribe")
            return

        # Maintain buffer over last 100ms for VAD check
        # 16000 samples/second * 0.1 seconds * 2 bytes/sample 
        max_bytes = 1600 * 2
        self.audio_bytes_buffer = (self.audio_bytes_buffer + audio_bytes)[max_bytes:]

        if len(self.audio_bytes_buffer) > len(audio_bytes):
            # VAD model isn't perfect, but it's assumed that checking over 100ms will be accurate enough for active speech
            contains_speech = self.vad_check(self.audio_bytes_buffer)
        else:
            # Otherwise if we haven't transcribed in a while, there might be a longer period to check
            contains_speech = self.vad_check(audio_bytes)

        if not contains_speech:
            logger.debug("No speech detected")
            # If we have audio to transcribe, transcribe it now that speech has stopped
            if len(self.audio_arr) > 0:
                result = self.transcribe_arr()
                if result.text.strip():
                    await self.pubsub.publish(EventType.HEARD_SPEECH, result.text.strip())
                    record_result(self.audio_arr, result.text)
                # Reset audio array without any trailing audio, as there is unlikely to be any words which span to the next chunk
                self.audio_arr = np.array([])
            return
        
        # TODO: VAD check might be too sensitive
        await self.pubsub.publish(EventType.HEARD_SPEAKING, None)

        logger.debug("Speech detected - transcribing")

        # Add new audio to the mel buffer
        if not self.previous_contains_speech:
            # Include at least the full audio buffer as VAD detection is imperfect and 
            # previous chunk may be part of a speech segment
            audio_bytes = max(audio_bytes, self.audio_bytes_buffer, key=len)

        audio_arr = audio_bytes_to_np_array(audio_bytes)
        self.audio_arr = np.concatenate([self.audio_arr, audio_arr])
        self.previous_contains_speech = contains_speech

        # If within 1000ms of buffer being full, transcribe
        # We expect delay to transcription to be called in <1000ms intervals
        # TODO: run transcription in separate process so this isn't an issue
        if len(self.audio_arr) >= self.num_audio_arr_samples - 1600:
            result = self.transcribe_arr()
            if result.text.strip():
                await self.pubsub.publish(EventType.HEARD_SPEECH, result.text.strip())
                record_result(self.audio_arr, result.text)
            # Keep the last 500ms of audio in the buffer in case it's needed to understand the next chunk
            # Last 500ms as 16khz = 8000 samples
            self.audio_arr = self.audio_arr[-8000:]

        # Otherwise keep accumulating and return empty result for now
        return
        
    def transcribe_arr(self, max_tokens=20):
        start_time = time.time()

        audio_arr = self.audio_arr
        audio_arr = nr.reduce_noise(y=audio_arr, sr=SAMPLING_RATE)

        mel = audio.log_mel_spectrogram(audio_arr, self.whisper_mlx_model.dims.n_mels)
        mel = mel.reshape(1, *mel.shape)

        # Drop audio from beginning to ensure it fits in space
        if mel.shape[1] > self.num_mel_frames:
            mel = mel[:, -self.num_mel_frames:, :]

        # Add some empty frames to the end of the mel_arr to ensure that the model can predict the end of transcript
        total_frames = self.num_mel_frames + self.num_empty_buffer_frames
        input_mel = mx.concatenate([
                mel,
                mx.zeros((1, total_frames - mel.shape[1], self.whisper_mlx_model.dims.n_mels), dtype=mx.float32)
            ], axis=1)

        decoded_tokens = mx.array([self.tokenizer.sot_sequence_including_notimestamps], dtype=mx.int32)
        result_tokens = []
        kv_cache = None
        next_token = None

        audio_features = self.whisper_mlx_model.encoder(input_mel)

        for i in range (max_tokens):
            if i == 0:
                logits, kv_cache, _ = self.whisper_mlx_model.decoder(decoded_tokens, audio_features, kv_cache)
            else:
                logits, kv_cache, _ = self.whisper_mlx_model.decoder(next_token.reshape(1, -1), audio_features, kv_cache)
            next_token = mx.argmax(logits[:, -1], axis=-1)
            next_token_value = next_token.item()

            if i == 0 and next_token_value == self.tokenizer.no_speech:
                return TranscriptionResult(text="", tokens=[])

            if next_token_value == self.tokenizer.eot:
                break
            result_tokens.append(next_token_value)
            decoded_tokens = mx.concatenate([decoded_tokens, next_token.reshape(1, -1)], axis=-1)

        end_time = time.time()
        whisper_time_ms = int(1000 * (end_time - start_time))

        result = TranscriptionResult(
            text=self.tokenizer.decode(result_tokens) if result_tokens else "",
            tokens=result_tokens
        )

        logger.info(f"whisper: {whisper_time_ms}ms : {result.text}")

        return result

    def vad_check(self, audio_bytes):
        contains_speech = False
        # Window size is 160 = 10 ms at 16kHz * 2 bytes per sample
        window_size = 160 * 2
        for i in range(0, len(audio_bytes), window_size):
            contains_speech = contains_speech or self.vad.is_speech(audio_bytes[i:i+window_size], SAMPLING_RATE)
        return contains_speech

