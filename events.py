import asyncio

from enum import Enum, auto

class EventType(Enum):
    HEARD_AUDIO = auto()
    HEARD_SPEECH = auto()
    HEARD_SPEAKING = auto()
    HEARD_PAUSE = auto()
    HEARD_NO_SPEAKING = auto()
    RESPONSE_TEXT_GENERATED = auto()
    RESPONSE_SPEECH_GENERATED = auto()

class PubSub:
    def __init__(self):
        self.subscribers = {}

    async def publish(self, event_type, data):
        if event_type in self.subscribers:
            for _, callback in self.subscribers[event_type]:
                await callback(data)

    def subscribe(self, event_type, callback, priority=5):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append((priority, callback))
        self.subscribers[event_type].sort(key=lambda x: x[0])

    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers:
            self.subscribers[event_type] = [(p, c) for p, c in self.subscribers[event_type] if c != callback]
