"""
Emotion System for Singularis AGI

Provides emotion and emotional valence emulation using HuiHui model.
Runs in parallel with all other AGI systems.
"""

from .huihui_emotion import (
    HuiHuiEmotionEngine,
    EmotionState,
    EmotionalValence,
    EmotionType,
    EmotionConfig
)

__all__ = [
    'HuiHuiEmotionEngine',
    'EmotionState',
    'EmotionalValence',
    'EmotionType',
    'EmotionConfig'
]
