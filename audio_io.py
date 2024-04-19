import multiprocessing as mp
import pyaudio
import time
from events import EventType
import asyncio
import logging

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 160 # 10ms
CHANNELS = 1

def input_audio_stream_process(audio_queue, device_name_like=None):
    print(f'Finding speech using device_name_like={device_name_like}')
    device_idx = find_device_idx(device_name_like)
    print(f'Finding speech using device_idx={device_idx}')

    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        audio_queue.put((time.time(), time_info, in_data))
        return (None, pyaudio.paContinue)

    print('Starting input stream')

    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=device_idx,
        stream_callback=callback)

    while stream.is_active():
        time.sleep(0.1)

    print('Stopping input stream')

    stream.close()
    p.terminate()

def output_audio_stream_process(audio_queue):
    print('Starting output stream')
    try:
        pya = pyaudio.PyAudio()
        stream = pya.open(format=pyaudio.paFloat32,
                                    channels=CHANNELS,
                                    rate=SAMPLE_RATE,
                                    output=True)

        while True:
            print('Checking for audio to play')
            speech_arr = audio_queue.get()
            if speech_arr is None:
                break
            print('Playing speech')
            stream.write(speech_arr.tobytes())

        stream.stop_stream()
        stream.close()
        pya.terminate()
    except Exception as e:
        print(f"Exception in play_speech_process: {str(e)}")

def find_device_idx(device_name_like=None):
    p = pyaudio.PyAudio()

    if not device_name_like:
        return p.get_default_input_device_info()['index']

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if device_name_like in info['name']:
            print(f'Found device "{info["name"]}" with index {i}')
            return i

    return p.get_default_input_device_info()['index']

class AudioIO():

    def __init__(self, pubsub, device_name_like=None, publish_interval_ms=50):
        self.pubsub = pubsub
        self.device_name_like = device_name_like
        self.publish_interval_ms = publish_interval_ms
        self.input_audio_process = None
        self.output_audio_process = None

    async def start(self):
        logger.info('Starting audio input')
        ctx = mp.get_context('spawn')
        self.input_audio_queue = ctx.Queue()
        self.output_audio_queue = ctx.Queue()
        self.input_audio_process = ctx.Process(
            target=input_audio_stream_process, args=(self.input_audio_queue, self.device_name_like, ))
        self.input_audio_process.start()
        self.output_audio_process = ctx.Process(
            target=output_audio_stream_process, args=(self.output_audio_queue, ))
        self.output_audio_process.start()

        while True:
            # TODO: add echo cancellation here
            latest_audio = self._get_latest_audio()
            logger.debug(f"Publishing {len(latest_audio)} bytes of audio")
            await self.pubsub.publish(EventType.HEARD_AUDIO, latest_audio)
            # TODO: adjust sleep time based upon how long it the loop took
            await asyncio.sleep(self.publish_interval_ms / 1000)

    def stop(self):
        if self.input_audio_process is not None:
            self.input_audio_process.terminate()
            self.input_audio_process = None
        if self.output_audio_process is not None:
            self.output_audio_process.terminate()
            self.output_audio_process = None

    def play_output_audio(self, speech_arr):
        logger.debug(f'Playing audio of length {len(speech_arr)}')
        chunk_size = 1600  # 100ms chunk at 16000Hz
        for i in range(0, len(speech_arr), chunk_size):
            self.output_audio_queue.put(speech_arr[i:i+chunk_size])

    def stop_playing_audio(self):
        logger.debug(f'Stopping audio playback')
        count = 0
        while True:
            try:
                self.output_audio_queue.get(block=False)
                count += 1
            except:
                break
        logger.debug(f'Removed {count} chunks from output queue')

    def _get_latest_audio(self):
        chunk_buffer = []
        while not self.input_audio_queue.empty():
            chunk_buffer.append(self.input_audio_queue.get())
        concatenated_bytes = b''.join(chunk[2] for chunk in chunk_buffer)
        return concatenated_bytes