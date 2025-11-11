"""
Skyrim-Specific Cognition System

This module defines game-specific cognitive dimensions and evaluation metrics
for autonomous Skyrim gameplay, replacing abstract philosophical concepts
with concrete game mechanics.

Key Cognitive Dimensions for Skyrim:
1. Survival: Health, stamina, safety from threats
2. Progression: Skills, levels, quest completion
3. Resources: Gold, items, equipment quality
4. Knowledge: Map exploration, NPC relationships, game mechanics
5. Effectiveness: Combat success, stealth success, social success
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List
import numpy as np


class CognitiveDimension(Enum):
    """Core cognitive dimensions for Skyrim gameplay evaluation."""
    SURVIVAL = "survival"  # Health, safety, threat avoidance
    PROGRESSION = "progression"  # Skills, levels, quest completion
    RESOURCES = "resources"  # Gold, items, equipment
    KNOWLEDGE = "knowledge"  # Exploration, NPC info, mechanics
    EFFECTIVENESS = "effectiveness"  # Combat, stealth, social success


@dataclass
class SkyrimCognitiveState:
    """
    Represents the agent's cognitive evaluation of the current game state.
    
    This replaces abstract "coherence" with concrete game metrics.
    """
    # Core dimensions (0.0 to 1.0 normalized)
    survival: float  # 1.0 = full health/safe, 0.0 = critical danger
    progression: float  # 1.0 = many skills/quests, 0.0 = no progress
    resources: float  # 1.0 = well-equipped/rich, 0.0 = poor/weak gear
    knowledge: float  # 1.0 = explored/informed, 0.0 = lost/ignorant
    effectiveness: float  # 1.0 = mastery, 0.0 = ineffective
    
    # Overall game state quality (weighted average)
    overall_quality: float
    
    @classmethod
    def from_game_state(cls, state: Dict[str, Any]) -> 'SkyrimCognitiveState':
        """
        Compute cognitive evaluation from raw game state.
        
        Args:
            state: Raw game state dict with health, gold, skills, etc.
            
        Returns:
            SkyrimCognitiveState with normalized evaluations
        """
        # Survival: Based on health ratio and threat level
        health_ratio = state.get('health', 50) / state.get('max_health', 100)
        in_combat = state.get('in_combat', False)
        enemy_nearby = state.get('enemy_nearby', False)
        survival = health_ratio * (0.5 if in_combat else 1.0) * (0.7 if enemy_nearby else 1.0)
        survival = np.clip(survival, 0.0, 1.0)
        
        # Progression: Based on skills and level
        avg_skill = state.get('average_skill_level', 15) / 100.0  # Skills go 0-100
        player_level = state.get('player_level', 1) / 81.0  # Max level ~81
        quest_count = state.get('completed_quests', 0) / 100.0  # Normalize to ~100 quests
        progression = (avg_skill * 0.4 + player_level * 0.3 + quest_count * 0.3)
        progression = np.clip(progression, 0.0, 1.0)
        
        # Resources: Based on gold and equipment
        gold_score = min(state.get('gold', 0) / 10000.0, 1.0)  # 10k gold = good
        equipment_quality = state.get('equipment_quality', 0.3)  # 0-1 scale
        carry_weight_ratio = state.get('carry_weight', 0) / state.get('max_carry_weight', 300)
        resources = (gold_score * 0.4 + equipment_quality * 0.4 + 
                    (1.0 - carry_weight_ratio) * 0.2)  # Less weight = more room
        resources = np.clip(resources, 0.0, 1.0)
        
        # Knowledge: Based on exploration and NPC interactions
        locations_discovered = state.get('locations_discovered', 0) / 343.0  # 343 locations
        npcs_met = state.get('npcs_met', 0) / 100.0  # Normalize to ~100 important NPCs
        mechanics_learned = state.get('mechanics_learned', 0) / 50.0  # ~50 mechanics
        knowledge = (locations_discovered * 0.4 + npcs_met * 0.3 + mechanics_learned * 0.3)
        knowledge = np.clip(knowledge, 0.0, 1.0)
        
        # Effectiveness: Based on combat/stealth/social success
        combat_success = state.get('combat_win_rate', 0.5)
        stealth_success = state.get('stealth_success_rate', 0.5)
        social_success = state.get('persuasion_success_rate', 0.5)
        effectiveness = (combat_success * 0.4 + stealth_success * 0.3 + social_success * 0.3)
        effectiveness = np.clip(effectiveness, 0.0, 1.0)
        
        # Overall quality: Weighted average with survival most important
        overall = (
            survival * 0.30 +  # Staying alive is critical
            progression * 0.25 +  # Character growth matters
            resources * 0.15 +  # Gear and gold help
            knowledge * 0.15 +  # Understanding the world
            effectiveness * 0.15  # Skill at tasks
        )
        
        return cls(
            survival=survival,
            progression=progression,
            resources=resources,
            knowledge=knowledge,
            effectiveness=effectiveness,
            overall_quality=overall
        )
    
    def quality_change(self, previous: 'SkyrimCognitiveState') -> float:
        """
        Calculate change in overall game state quality.
        
        This replaces the philosophical "Δ𝒞" (coherence change) with
        concrete game state improvement.
        
        Args:
            previous: Previous cognitive state
            
        Returns:
            Change in quality (-1.0 to 1.0)
        """
        return self.overall_quality - previous.overall_quality
    
    def dimension_changes(self, previous: 'SkyrimCognitiveState') -> Dict[str, float]:
        """Get changes in each cognitive dimension."""
        return {
            'survival': self.survival - previous.survival,
            'progression': self.progression - previous.progression,
            'resources': self.resources - previous.resources,
            'knowledge': self.knowledge - previous.knowledge,
            'effectiveness': self.effectiveness - previous.effectiveness,
        }


class SkyrimActionEvaluator:
    """
    Evaluates actions based on game-specific criteria.
    
    This replaces philosophical "ethical evaluation" (Δ𝒞 > 0)
    with practical game evaluation (improves game state?).
    """
    
    @staticmethod
    def evaluate_action_outcome(
        state_before: SkyrimCognitiveState,
        state_after: SkyrimCognitiveState,
        action: str
    ) -> Dict[str, Any]:
        """
        Evaluate whether an action improved the game state.
        
        Args:
            state_before: State before action
            state_after: State after action
            action: Action taken
            
        Returns:
            Dict with evaluation results
        """
        quality_change = state_after.quality_change(state_before)
        dimension_changes = state_after.dimension_changes(state_before)
        
        # Determine overall assessment
        if quality_change > 0.05:
            assessment = "BENEFICIAL"  # Action clearly helped
        elif quality_change < -0.05:
            assessment = "DETRIMENTAL"  # Action hurt us
        else:
            assessment = "NEUTRAL"  # Minimal impact
        
        # Identify which dimensions improved/degraded
        improved = [dim for dim, change in dimension_changes.items() if change > 0.02]
        degraded = [dim for dim, change in dimension_changes.items() if change < -0.02]
        
        return {
            'quality_change': quality_change,
            'assessment': assessment,
            'dimension_changes': dimension_changes,
            'improved_dimensions': improved,
            'degraded_dimensions': degraded,
            'action': action
        }


class SkyrimMotivation:
    """
    Game-specific motivation system.
    
    This replaces abstract "intrinsic motivation" (curiosity, coherence, etc.)
    with concrete game motivations.
    """
    
    def __init__(
        self,
        survival_weight: float = 0.35,
        progression_weight: float = 0.25,
        exploration_weight: float = 0.20,
        wealth_weight: float = 0.10,
        mastery_weight: float = 0.10
    ):
        """
        Initialize with motivation weights.
        
        Args:
            survival_weight: How much to value staying alive
            progression_weight: How much to value leveling/skills
            exploration_weight: How much to value discovering new areas
            wealth_weight: How much to value gold/loot
            mastery_weight: How much to value combat/stealth mastery
        """
        self.survival_weight = survival_weight
        self.progression_weight = progression_weight
        self.exploration_weight = exploration_weight
        self.wealth_weight = wealth_weight
        self.mastery_weight = mastery_weight
    
    def compute_motivation_score(
        self,
        action: str,
        current_state: SkyrimCognitiveState,
        context: Dict[str, Any]
    ) -> float:
        """
        Compute how motivated the agent is to take this action.
        
        Args:
            action: Action to evaluate
            current_state: Current cognitive state
            context: Additional context (threats, opportunities, etc.)
            
        Returns:
            Motivation score (0.0 to 1.0)
        """
        score = 0.0
        
        # Survival motivation: High when in danger
        if action in ['retreat', 'heal', 'block', 'dodge']:
            if current_state.survival < 0.5:
                score += self.survival_weight * (1.0 - current_state.survival)
        
        # Progression motivation: High when can gain XP/skills
        if action in ['accept_quest', 'complete_quest', 'practice_skill', 'level_up']:
            if current_state.progression < 0.7:
                score += self.progression_weight * (1.0 - current_state.progression)
        
        # Exploration motivation: High in unexplored areas
        if action in ['explore', 'move_forward', 'investigate']:
            unexplored = context.get('unexplored_nearby', False)
            if unexplored and current_state.knowledge < 0.6:
                score += self.exploration_weight * (1.0 - current_state.knowledge)
        
        # Wealth motivation: High when can get loot
        if action in ['loot', 'take_item', 'pickpocket', 'sell']:
            valuable_nearby = context.get('valuable_nearby', False)
            if valuable_nearby and current_state.resources < 0.6:
                score += self.wealth_weight * (1.0 - current_state.resources)
        
        # Mastery motivation: High when can improve skills
        if action in ['attack', 'cast_spell', 'sneak', 'persuade']:
            practice_opportunity = context.get('practice_opportunity', False)
            if practice_opportunity and current_state.effectiveness < 0.7:
                score += self.mastery_weight * (1.0 - current_state.effectiveness)
        
        return np.clip(score, 0.0, 1.0)
    
    def get_dominant_motivation(self, current_state: SkyrimCognitiveState) -> str:
        """
        Determine the currently dominant motivation.
        
        Args:
            current_state: Current cognitive state
            
        Returns:
            Name of dominant motivation
        """
        # Survival always dominant when in danger
        if current_state.survival < 0.4:
            return "SURVIVAL"
        
        # Otherwise, check what's lacking
        motivations = {
            'PROGRESSION': self.progression_weight * (1.0 - current_state.progression),
            'EXPLORATION': self.exploration_weight * (1.0 - current_state.knowledge),
            'WEALTH': self.wealth_weight * (1.0 - current_state.resources),
            'MASTERY': self.mastery_weight * (1.0 - current_state.effectiveness),
        }
        
        return max(motivations.items(), key=lambda x: x[1])[0]


def compute_action_value(
    action: str,
    cognitive_state: SkyrimCognitiveState,
    motivation: SkyrimMotivation,
    context: Dict[str, Any]
) -> float:
    """
    Compute overall value of an action combining multiple factors.
    
    This is a game-specific replacement for philosophical action evaluation.
    
    Args:
        action: Action to evaluate
        cognitive_state: Current cognitive state
        motivation: Motivation system
        context: Additional context
        
    Returns:
        Action value score (0.0 to 1.0)
    """
    # Get motivation-based score
    motivation_score = motivation.compute_motivation_score(
        action, cognitive_state, context
    )
    
    # Add tactical considerations
    tactical_bonus = 0.0
    
    # Prefer safe actions when health is low
    if cognitive_state.survival < 0.4:
        if action in ['retreat', 'heal', 'block']:
            tactical_bonus += 0.3
        elif action in ['attack', 'advance']:
            tactical_bonus -= 0.3
    
    # Prefer learning actions when effectiveness is low
    if cognitive_state.effectiveness < 0.5:
        if action in ['practice_skill', 'train', 'experiment']:
            tactical_bonus += 0.2
    
    # Prefer resource gathering when poor
    if cognitive_state.resources < 0.3:
        if action in ['loot', 'sell', 'gather']:
            tactical_bonus += 0.2
    
    # Combine scores
    final_score = motivation_score + tactical_bonus
    return np.clip(final_score, 0.0, 1.0)
