import argparse
import asyncio
import logging
import time

from audio_io import AudioIO
from incremental_transcriber import IncrementalTranscriber
from speech import SpeechGenerator
from llm_chat import VoiceChatAgent
from whisper_mlx.whisper_mlx import load_model as load_whisper_model
from chat_model import load_chat_model
from events import PubSub, EventType

logger = logging.getLogger(__name__)

class EventTimer:

    def __init__(self, pubsub):
        self.pubsub = pubsub
        self.last_heard_speech_time_ms = None
        self.pubsub.subscribe(EventType.HEARD_SPEAKING, self.handle_heard_speaking, priority=0)
        self.pubsub.subscribe(EventType.RESPONSE_SPEECH_GENERATED, self.handle_response_speech_generated, priority=0)
    
    async def handle_heard_speaking(self, event_data):
        self.last_heard_speech_time_ms = int(time.time() * 1000)

    async def handle_response_speech_generated(self, event_data):
        if self.last_heard_speech_time_ms is not None:
            speech_to_speech_ms = int(time.time() * 1000) - self.last_heard_speech_time_ms
            logger.info(f"Speech to speech ms: {speech_to_speech_ms}")
            # Only want to log this for the first chunk of response speech
            self.last_heard_speech_time_ms = None

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio file.')
    parser.add_argument('--whisper_mlx_model_dir', type=str, help='Path to the whisper mlx model.')
    parser.add_argument('--mlxlm_model_path', type=str, help='Path to the mlx lm model.')
    parser.add_argument('--device_name', type=str, default=None, help='Name for the input audio device.')
    parser.add_argument('--log_dir', type=str, default="logs", help='Directory for speech and chat logs.')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level')
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info("Loading Models ...")
    chat_model = load_chat_model(args.mlxlm_model_path, args.log_dir)
    whisper_model = load_whisper_model(args.whisper_mlx_model_dir)

    logging.info("Initializing ...")
    pubsub = PubSub()
    event_timer = EventTimer(pubsub)
    audio_io = AudioIO(pubsub, device_name_like=args.device_name)
    transcriber = IncrementalTranscriber(pubsub, whisper_model, args.log_dir)
    speech_generator = SpeechGenerator(pubsub, audio_io)
    agent = VoiceChatAgent(pubsub, chat_model)
    
    logging.info("Starting input audio ...")
    await audio_io.start()
   
if __name__ == '__main__':
    asyncio.run(main())
