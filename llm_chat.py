import asyncio
from asyncio import Queue
import time
import numpy as np
import mlx.core as mx
from enum import Enum, auto

from events import EventType

import logging

logger = logging.getLogger(__name__)

class AgentState(Enum):
    WAITING = auto()
    LISTENING = auto()
    RESPONDING = auto()

INIT_PROMPT = """You are a voice assistant named Zoe. You hear short chunks of audio with possibly small overlaps.
Follow these rules:
* The audio transcription might contain errors, so guess what the user is saying.
* Generate a response only if needed.
* Prefer short responses
* Include commas in your responses.
* Use a spoken style response rather than a written style response, use commas to indicate a break in speech.

"""

INIT_EVENTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the time?", "Sorry, I don't know."),
    ("What is the capital of Germany?", "Berlin"),
    ("Stop", ""),
    ("How long would it take to", ""),
    ("How long would it take to walk from Paris to London", "Sorry, I don't know."),
    ("What is your name?", "Zoe")
]

NUM_TOKENS_RESPONSE_PAUSE = 5
END_OF_USER_PROBABILITY_THRESHOLD = 0.5

class VoiceChatAgent:

    def __init__(self, pubsub, chat_model):
        self.pubsub = pubsub
        self.chat_model = chat_model

        self.event_queue = Queue()
        self.is_running = False
        self.process_events_task = None
        self.response_task = None
        self.last_user_query = ""

        self.chat_model.initialize(INIT_PROMPT, INIT_EVENTS)
        self.state = AgentState.WAITING
        self.is_heard_speaking = False
        self.pubsub.subscribe(EventType.HEARD_SPEECH, lambda data: self.handle_event(EventType.HEARD_SPEECH, data))
        self.pubsub.subscribe(EventType.HEARD_SPEAKING, lambda data: self.handle_event(EventType.HEARD_SPEAKING, None))
        self.pubsub.subscribe(EventType.HEARD_NO_SPEAKING, lambda data: self.handle_event(EventType.HEARD_NO_SPEAKING, None))

    async def handle_event(self, event_type, data):
        logger.debug(f"Handling event {event_type} with data: {data}")
        await self.event_queue.put((event_type, data))
        if not self.is_running:
            logger.debug("Starting process_events task")
            self.is_running = True
            self.process_events_task = asyncio.create_task(self.process_events())
        else:
            logger.debug("process_events already running")

    async def process_events(self):
        while self.is_running:
            event = await self.event_queue.get()
            logger.debug(f"Processing event: {event}, event type: {event[0]}, data: {event[1]}")
            
            try:
                if event[0] == EventType.HEARD_SPEECH:
                    if event[1].strip() == "":
                        continue

                    # Cancel any existing response task
                    if self.response_task and not self.response_task.done():
                        logger.debug("Heard new speech while responding, cancelling response task")
                        self.response_task.cancel()
                        try:
                            # Wait for the task to be cancelled.
                            # This ensures any cleanup or finalization can occur within the task.
                            await self.response_task
                        except asyncio.CancelledError:
                            logger.debug("Response generation task was cancelled.")
                        finally:
                            self.response_task = None  # Reset the task to None after cancellation

                    self._update_heard_speech(event[1])

                elif event[0] == EventType.HEARD_SPEAKING:
                    self.is_heard_speaking = True
                
                elif event[0] == EventType.HEARD_NO_SPEAKING:
                    if self.state == AgentState.WAITING:
                        logger.debug("Paused while waiting for user input, ignoring")
                        continue

                    self.is_heard_speaking = False
                    if self.response_task is None or self.response_task.done():
                        logger.debug("Creating new response task")
                        self.response_task = asyncio.create_task(self._generate_response())
                    else:
                        logger.debug("Response task already running")
            except Exception as e:
                logger.error(f"Error processing event {event[0]}: {str(e)}")

    def _update_heard_speech(self, transcript):
        logger.debug(f"LLM handling heard speech: {transcript}")

        if self.state == AgentState.RESPONDING:
            logger.debug("Interruption detected")

        logger.debug(f"In state {self.state}, switching to LISTENING")
        self.state = AgentState.LISTENING

        self.chat_model.add_user_message_segment(transcript)
        self.last_user_query += transcript + " "

    async def _generate_response(self):
        logger.debug("Generating response")

        next_response_token = None
        response_tokens_so_far = []

        while True:
            # Pause to let other tasks process
            await asyncio.sleep(0)

            if self.is_heard_speaking:
                # Wait to see if new speech is heard, response will be cancelled in that case
                # TODO: this could be based upon a pause flag
                logger.debug("Heard speaking while generating response - waiting for pause")
                await asyncio.sleep(0)
                logger.debug("Resuming response generation after pause for user speaking")
                continue

            end_user_prob = self.chat_model.prob_end_of_user_message()
            if end_user_prob < END_OF_USER_PROBABILITY_THRESHOLD:
                # If we aren't confident user has finished their message, wait before responding
                # They might say more after they have paused for a few seconds
                logger.debug(f"Prob end of user message {end_user_prob} below threshold, switching back to WAITING")
                self.state = AgentState.WAITING
                return

            logger.debug(f"In state {self.state}, switching to RESPONDING")
            self.state = AgentState.RESPONDING

            # TODO: run tool model using self.last_user_query
            
            self.last_user_query = ""

            response_segment_result = self.chat_model.generate_response_segment(
                response_tokens_so_far, next_response_token, num_tokens=NUM_TOKENS_RESPONSE_PAUSE)

            next_response_token = response_segment_result.next_response_token
            response_tokens_so_far = response_segment_result.tokens

            if response_segment_result.say_segment:
                logger.info(f"Speaking response: {response_segment_result.text}")
                response_tokens_so_far = []
                await self.pubsub.publish(EventType.RESPONSE_TEXT_GENERATED, response_segment_result.text)

            if response_segment_result.is_complete:
                logger.debug(f"In state {self.state}, switching to WAITING")
                self.state = AgentState.WAITING
                return

            # Pausing response to check for new audio,
            # and will keep generating if there are no interuptions
            logger.debug(f"Pausing to check for interuptions with partial response: {response_segment_result.text}")
