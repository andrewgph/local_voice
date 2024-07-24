import asyncio
from asyncio import Queue
import numpy as np
from webrtcvad import Vad

from events import EventType
from audio_io import SAMPLE_RATE

import logging

logger = logging.getLogger(__name__)


# Window size is 160 = 10 ms at 16kHz * 2 bytes per sample
AUDIO_BYTES_WINDOW_SIZE = 160 * 2 * 3
# Check last 100ms for VAD to identify pauses
AUDIO_BYTES_VAD_CHECK_SIZE = AUDIO_BYTES_WINDOW_SIZE * 3


def audio_bytes_to_np_array(bytes_data):
    arr = np.frombuffer(bytes_data, dtype='<i2')
    arr = arr.astype('float32') / 32768.0
    return arr


def vad_check(vad, audio_bytes):
    window_count = 0
    speech_count = 0
    for i in range(0, len(audio_bytes), AUDIO_BYTES_WINDOW_SIZE):
        window_count += 1
        if vad.is_speech(audio_bytes[i:i+AUDIO_BYTES_WINDOW_SIZE], SAMPLE_RATE):
            speech_count += 1
    logger.debug(f"VAD found speech in {speech_count} / {window_count} windows")
    return speech_count > 0


class VADChecker:

    def __init__(self, pubsub):
        self.audio_bytes_vad_buffer = b''
        self.audio_bytes_buffer = b''
        self.vad = Vad()
        self.vad.set_mode(2)

        self.event_queue = Queue()
        self.is_running = False
        self.vad_task = None

        self.pubsub = pubsub
        self.pubsub.subscribe(EventType.HEARD_AUDIO, self.handle_heard_audio, priority=0)

    async def handle_heard_audio(self, audio_bytes):
        logger.debug(f"Handling heard audio {len(audio_bytes)} bytes")
        await self.event_queue.put(audio_bytes)
        if not self.is_running:
            logger.debug("Starting VAD checker")
            self.is_running = True
            self.transcribe_task = asyncio.create_task(self.vad_check())
        else:
            logger.debug("VAD checker already running")

    async def vad_check(self):
        while self.is_running:
            audio_bytes = await self.event_queue.get()

            logger.debug(f"VAD check {len(audio_bytes)} bytes")

            if len(audio_bytes) == 0:
                logger.debug("No audio to check")
                continue

            # Fill up the VAD check buffer
            self.audio_bytes_vad_buffer += audio_bytes
            if len(self.audio_bytes_vad_buffer) < AUDIO_BYTES_VAD_CHECK_SIZE:
                logger.debug("Waiting for VAD check buffer to fill up")
                continue

            logger.debug(f"Audio bytes VAD check buffer: {len(self.audio_bytes_vad_buffer)} bytes")

            if vad_check(self.vad, self.audio_bytes_vad_buffer):
                logger.debug("Recent speech detected by VAD")
                await self.pubsub.publish(EventType.HEARD_SPEAKING, self.audio_bytes_vad_buffer)
            else:
                logger.debug("No recent speech detected by VAD")
                await self.pubsub.publish(EventType.HEARD_NO_SPEAKING, None)

            # Reset VAD check buffer
            self.audio_bytes_vad_buffer = b''