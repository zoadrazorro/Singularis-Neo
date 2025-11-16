"""
Sophia Productivity Module

Intelligent task and calendar management powered by LifeOps AGI.
"""

from .sync_cache import SyncCache
from .suggestion_engine import SuggestionEngine, Suggestion, SuggestionType

__all__ = [
    'SyncCache',
    'SuggestionEngine',
    'Suggestion',
    'SuggestionType',
]
