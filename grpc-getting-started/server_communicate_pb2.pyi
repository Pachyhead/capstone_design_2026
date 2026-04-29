from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SpeechUploadRequest(_message.Message):
    __slots__ = ("sender_id", "receiver_id", "text", "emotion_vector")
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    EMOTION_VECTOR_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    receiver_id: str
    text: str
    emotion_vector: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, sender_id: _Optional[str] = ..., receiver_id: _Optional[str] = ..., text: _Optional[str] = ..., emotion_vector: _Optional[_Iterable[float]] = ...) -> None: ...

class UploadStatus(_message.Message):
    __slots__ = ("accepted",)
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    def __init__(self, accepted: bool = ...) -> None: ...

class UserIdentifier(_message.Message):
    __slots__ = ("user_id",)
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    def __init__(self, user_id: _Optional[str] = ...) -> None: ...

class AudioFrame(_message.Message):
    __slots__ = ("audio_content", "sender_id", "message_id", "is_final")
    AUDIO_CONTENT_FIELD_NUMBER: _ClassVar[int]
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    audio_content: bytes
    sender_id: str
    message_id: str
    is_final: bool
    def __init__(self, audio_content: _Optional[bytes] = ..., sender_id: _Optional[str] = ..., message_id: _Optional[str] = ..., is_final: bool = ...) -> None: ...
