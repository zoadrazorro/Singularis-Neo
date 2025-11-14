"""Learning module for text processing and knowledge integration."""

from .text_processor import (
    TextProcessor,
    TextChunk,
    CurriculumLoader,
    LearningProgress,
)
from .continual_learner import ContinualLearner
from .compositional_knowledge import CompositionalKnowledgeBuilder
from .curriculum_integration import CurriculumIntegration
from .curriculum_reward import CurriculumRewardFunction, CurriculumStage
from .curriculum_symbolic import CurriculumSymbolicRules

__all__ = [
    "TextProcessor",
    "TextChunk",
    "CurriculumLoader",
    "LearningProgress",
    "ContinualLearner",
    "CompositionalKnowledgeBuilder",
    "CurriculumIntegration",
    "CurriculumRewardFunction",
    "CurriculumStage",
    "CurriculumSymbolicRules",
]
