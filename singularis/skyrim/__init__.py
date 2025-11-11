"""
Skyrim Integration for Singularis AGI

Enables the Singularis AGI engine to play Skyrim autonomously.

This integration demonstrates:
- Multimodal perception (screen capture + CLIP vision)
- Real-time action control (keyboard/mouse or game API)
- Autonomous gameplay with intrinsic motivation
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
- skyrim_agi.py: Main AGI integration

Philosophical grounding:
- ETHICA: Conatus (∇𝒞) drives autonomous exploration
- Agency emerges from intrinsic motivation
- Learning is continual (no resets between episodes)
- Ethical choices measured by Δ𝒞 (coherence change)
"""

from .perception import SkyrimPerception, GameState, SceneType
from .actions import SkyrimActions, ActionType
from .skyrim_world_model import SkyrimWorldModel
from .skyrim_agi import SkyrimAGI, SkyrimConfig

__all__ = [
    'SkyrimPerception',
    'GameState',
    'SceneType',
    'SkyrimActions',
    'ActionType',
    'SkyrimWorldModel',
    'SkyrimAGI',
    'SkyrimConfig',
]
