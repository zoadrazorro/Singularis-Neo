"""
Skyrim Integration for Singularis AGI

Enables the Singularis AGI engine to play Skyrim autonomously.

This integration demonstrates:
- Multimodal perception (screen capture + CLIP vision)
- Real-time action control (keyboard/mouse or game API)
- Autonomous gameplay with game-specific motivation
- Causal learning from gameplay experience
- Long-term skill development and goal formation

Why Skyrim?
- Massive complexity: 300+ locations, 1000+ NPCs, 10,000+ items
- Open-ended: No single goal, emergent objectives
- Long-term: Thousands of hours of content
- Rich narrative: Quests with moral choices and consequences

Components:
- perception.py: Screen capture, CLIP vision, game state reading
- actions.py: Action control (keyboard/mouse, game API)
- skyrim_world_model.py: Skyrim-specific world model extensions
- skyrim_cognition.py: Game-specific evaluation and motivation
- skyrim_agi.py: Main AGI integration

Design principles:
- Game-specific cognition replaces abstract philosophical concepts
- Agent learns what works through reinforcement learning
- Decisions based on survival, progression, and effectiveness
- Motivation driven by game goals (not abstract coherence)
"""

from .perception import SkyrimPerception, GameState, SceneType
from .actions import SkyrimActions, ActionType
from .skyrim_world_model import SkyrimWorldModel
from .skyrim_cognition import SkyrimCognitiveState, SkyrimMotivation, SkyrimActionEvaluator
from .skyrim_agi import SkyrimAGI, SkyrimConfig

__all__ = [
    'SkyrimPerception',
    'GameState',
    'SceneType',
    'SkyrimActions',
    'ActionType',
    'SkyrimWorldModel',
    'SkyrimCognitiveState',
    'SkyrimMotivation',
    'SkyrimActionEvaluator',
    'SkyrimAGI',
    'SkyrimConfig',
]
