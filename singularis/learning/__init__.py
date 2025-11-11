"""Learning module for text processing and knowledge integration."""

from .text_processor import (
    TextProcessor,
    TextChunk,
    CurriculumLoader,
    LearningProgress,
)
from .continual_learner import ContinualLearner
from .compositional_knowledge import CompositionalKnowledgeBuilder

__all__ = [
    "TextProcessor",
    "TextChunk",
    "CurriculumLoader",
    "LearningProgress",
    "ContinualLearner",
    "CompositionalKnowledgeBuilder",
]
