import numpy as np
import torch
import pyaudio
import multiprocessing
import time

from events import EventType

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

import logging

logger = logging.getLogger(__name__)

class SpeechGenerator:

    def __init__(self, pubsub, audio_io):
        checkpoint = "microsoft/speecht5_tts"
        self.processor = SpeechT5Processor.from_pretrained(checkpoint)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.speaker_embedding = torch.tensor(np.load("/Users/andrew/Downloads/cmu_us_slt_arctic-wav-arctic_a0508.npy")).unsqueeze(0)
        
        self.pubsub = pubsub
        self.pubsub.subscribe(EventType.HEARD_SPEAKING, self.handle_heard_speaking)
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

    async def handle_heard_speaking(self, data):
        logger.debug(f"Handling heard speaking - stopping any speech")
        if self.is_generating_speech:
            self.discard_generated_speech = True
        self.stop_speaking()

    def generate_speech(self, text):
        start_time = time.time()
        speech_arr = self._predict(text)
        logger.info(f"Speech generation took {int((time.time() - start_time) * 1000)} milliseconds")    
        return speech_arr

    def _predict(self, text, speaker="SLT"):
        if len(text.strip()) == 0:
            return None

        start_process_time = time.time()
        inputs = self.processor(text=text, return_tensors="pt")
        logger.debug(f"Processing time: {int((time.time() - start_process_time) * 1000)} milliseconds")

        # limit input length
        input_ids = inputs["input_ids"]
        input_ids = input_ids[..., :self.model.config.max_text_positions]

        generate_speech_start_time = time.time()
        speech = self.model.generate_speech(input_ids, self.speaker_embedding, vocoder=self.vocoder)
        logger.debug(f"Generate speech time: {int((time.time() - generate_speech_start_time) * 1000)} milliseconds")

        return speech.numpy()

    def play_speech(self, speech_arr):
        logger.debug("Putting speech into audio queue")
        self.audio_io.play_output_audio(speech_arr)

    def stop_speaking(self):
        logger.debug("Stopping speaking")
        self.audio_io.stop_playing_audio()

