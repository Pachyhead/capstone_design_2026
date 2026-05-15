from dataclasses import dataclass
from enum import IntEnum

class EmotionLabel(IntEnum):
    ANGRY = 0,
    DISGUSTED = 1,
    FEARFUL = 2,
    HAPPY = 3,
    NEUTRAL = 4,
    OTHER = 5,
    SAD = 6,
    SURPRISED = 7,
    UNK = 8
    
@dataclass
class EncodeResult:
    text: str
    emotion_indices: tuple[int, ...]
    emotion_label: EmotionLabel
    emotion_score: float