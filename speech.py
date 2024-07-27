import numpy as np
import time
import librosa

from audio_io import SAMPLE_RATE
from min_rhasspy_piper.voice import PiperVoice

from events import EventType

import logging

logger = logging.getLogger(__name__)

class SpeechGenerator:

    def __init__(self, pubsub, audio_io, piper_model_dir):
        self.voice = PiperVoice.load(
            model_path=piper_model_dir + "/voice.onnx",
            config_path=piper_model_dir + "/voice.json"
        )
        
        self.pubsub = pubsub
        # Higher priority for heard speech so speech is interupted asap
        self.pubsub.subscribe(EventType.HEARD_SPEECH, self.handle_heard_speech, priority=1)
        self.pubsub.subscribe(EventType.RESPONSE_TEXT_GENERATED, self.handle_response_text_generated)
        self.is_generating_speech = False
        self.discard_generated_speech = False
        self.audio_io = audio_io

    async def handle_response_text_generated(self, response_text):
        logger.debug(f"Handling response text generated: {response_text}")
        
        self.is_generating_speech = True
        speech_arr = self.generate_speech(response_text)
        self.is_generating_speech = False
        
        if self.discard_generated_speech:
            logger.debug("Interupted - discarding generated speech")
            self.discard_generated_speech = False
        elif not speech_arr is None:
            self.play_speech(speech_arr)
        await self.pubsub.publish(EventType.RESPONSE_SPEECH_GENERATED, None)

    async def handle_heard_speech(self, data):
        logger.debug(f"Handling heard speech - stopping any speech")
        if self.is_generating_speech:
            self.discard_generated_speech = True
        self.stop_speaking()

    def generate_speech(self, text):
        start_time = time.time()
        speech_arr = self._predict(text)
        logger.info(f"Speech generation took {int((time.time() - start_time) * 1000)} milliseconds")    
        return speech_arr

    def _predict(self, text):
        if len(text.strip()) == 0:
            return None
        
        results = []
        for result in self.voice.synthesize_stream_raw(text):
            results.append(result)
        speech_arr = np.concatenate(results)

        logger.debug(f"Original speech array shape: {speech_arr.shape}")
        
        # Make the sample rate used in audio_io
        original_sr = self.voice.config.sample_rate
        target_sr = SAMPLE_RATE
        speech_arr = librosa.resample(speech_arr, orig_sr=original_sr, target_sr=target_sr)
        logger.debug(f"Resampled speech array from {original_sr} to {target_sr} shape: {speech_arr.shape}")

        return speech_arr

    def play_speech(self, speech_arr):
        logger.debug("Putting speech into audio queue")
        self.audio_io.play_output_audio(speech_arr)

    def stop_speaking(self):
        logger.debug("Stopping speaking")
        self.audio_io.stop_playing_audio()

