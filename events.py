import asyncio

from enum import Enum, auto

class EventType(Enum):
    HEARD_AUDIO = auto()
    HEARD_SPEECH = auto()
    HEARD_SPEAKING = auto()
    RESPONSE_TEXT_GENERATED = auto()
    RESPONSE_SPEECH_GENERATED = auto()

class PubSub:
    def __init__(self):
        self.subscribers = {}

    async def publish(self, event_type, data):
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                await callback(data)

    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)