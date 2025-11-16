"""
Action Integration - Connect Enhanced Actions to PersonModel

Integrates the enhanced action system with PersonModel scoring
to enable personality-driven, context-aware action selection.
"""

from typing import List, Dict, Optional, Tuple
from loguru import logger

from .enhanced_actions import EnhancedActionType, EnhancedAction, ActionCategory, get_affordance
from .action_affordance_system import ActionAffordanceSystem, GameContext, create_action_from_type
from ..person_model import PersonModel, score_action_for_person


class PersonalizedActionSelector:
    """
    Selects actions using PersonModel + Action Affordances.
    
    Combines:
    - Action affordances (what's possible)
    - PersonModel (personality, values, goals)
    - Context (game state, recent history)
    """
    
    def __init__(self, person: PersonModel):
        self.person = person
        self.affordance_system = ActionAffordanceSystem()
        
        logger.info(f"[ActionSelector] Initialized for {person.identity.name}")
    
    def update_context(self, game_state: Dict):
        """Update game context from state."""
        self.affordance_system.update_context(game_state)
    
    def select_action(
        self,
        being_state: any,
        base_score: float = 0.5,
        top_k: int = 5
    ) -> Optional[EnhancedAction]:
        """
        Select best action using PersonModel + affordances.
        
        Args:
            being_state: BeingState for context
            base_score: Base score for actions
            top_k: Consider top K available actions
        
        Returns:
            Selected EnhancedAction or None
        """
        # Get available actions
        available = self.affordance_system.get_available_actions()
        
        if not available:
            logger.warning(f"[ActionSelector] No available actions for {self.person.identity.name}")
            return None
        
        logger.debug(f"[ActionSelector] {len(available)} actions available")
        
        # Score all available actions
        scores = {}
        for action_type in available:
            # Create mock action for scoring
            action = create_action_from_type(action_type)
            
            # Score with PersonModel
            score = score_action_for_person(
                self.person,
                action,
                base_score=base_score
            )
            
            scores[action_type] = score
        
        # Filter invalid (negative scores)
        valid_scores = {a: s for a, s in scores.items() if s > -1e8}
        
        if not valid_scores:
            logger.warning(f"[ActionSelector] No valid actions after scoring")
            return None
        
        # Get top K
        top_actions = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Log top actions
        logger.info(f"[ActionSelector] Top {len(top_actions)} actions:")
        for action_type, score in top_actions:
            logger.info(f"  {action_type.value}: {score:.3f}")
        
        # Select best
        best_action_type, best_score = top_actions[0]
        
        # Create full action
        best_action = create_action_from_type(
            best_action_type,
            reason=f"Personality-driven (score={best_score:.2f})"
        )
        
        # Mark as executed
        self.affordance_system.execute_action(best_action)
        
        return best_action
    
    def select_action_for_goal(
        self,
        goal: str,
        being_state: any,
        base_score: float = 0.5
    ) -> Optional[EnhancedAction]:
        """
        Select action that helps achieve specific goal.
        
        Args:
            goal: Goal description (e.g., "escape", "attack")
            being_state: BeingState for context
            base_score: Base score for actions
        
        Returns:
            Selected EnhancedAction or None
        """
        # Get available actions
        available = self.affordance_system.get_available_actions()
        
        # Filter by goal
        goal_actions = self.affordance_system.get_actions_for_goal(goal, available)
        
        if not goal_actions:
            logger.warning(f"[ActionSelector] No actions available for goal: {goal}")
            return None
        
        logger.debug(f"[ActionSelector] {len(goal_actions)} actions for goal '{goal}'")
        
        # Score goal-relevant actions
        scores = {}
        for action_type in goal_actions:
            action = create_action_from_type(action_type)
            score = score_action_for_person(self.person, action, base_score)
            scores[action_type] = score
        
        # Select best
        valid_scores = {a: s for a, s in scores.items() if s > -1e8}
        if not valid_scores:
            return None
        
        best_action_type = max(valid_scores, key=valid_scores.get)
        best_action = create_action_from_type(
            best_action_type,
            reason=f"Goal: {goal} (score={valid_scores[best_action_type]:.2f})"
        )
        
        self.affordance_system.execute_action(best_action)
        return best_action
    
    def select_action_by_situation(
        self,
        being_state: any,
        base_score: float = 0.5
    ) -> Optional[EnhancedAction]:
        """
        Select action based on current situation.
        
        Automatically determines situation (offensive, defensive, etc.)
        and selects appropriate action.
        
        Args:
            being_state: BeingState for context
            base_score: Base score for actions
        
        Returns:
            Selected EnhancedAction or None
        """
        # Get available actions by situation
        situations = self.affordance_system.filter_by_situation()
        
        # Determine priority situation
        context = self.affordance_system.context
        
        # Emergency (low health)
        if context.health < 0.3 and situations['emergency']:
            logger.info("[ActionSelector] Emergency situation (low health)")
            return self._select_from_list(situations['emergency'], being_state, base_score, "emergency")
        
        # Combat (enemies nearby)
        if context.is_in_combat or context.num_enemies_nearby > 0:
            # Defensive if outnumbered or low health
            if context.num_enemies_nearby > 2 or context.health < 0.5:
                if situations['defensive']:
                    logger.info("[ActionSelector] Combat situation (defensive)")
                    return self._select_from_list(situations['defensive'], being_state, base_score, "defensive")
            
            # Offensive otherwise
            if situations['offensive']:
                logger.info("[ActionSelector] Combat situation (offensive)")
                return self._select_from_list(situations['offensive'], being_state, base_score, "offensive")
        
        # Stealth
        if context.is_sneaking and situations['stealth']:
            logger.info("[ActionSelector] Stealth situation")
            return self._select_from_list(situations['stealth'], being_state, base_score, "stealth")
        
        # Default: mobility/utility
        if situations['mobility']:
            logger.info("[ActionSelector] Exploration situation")
            return self._select_from_list(situations['mobility'], being_state, base_score, "exploration")
        
        return None
    
    def _select_from_list(
        self,
        action_types: List[EnhancedActionType],
        being_state: any,
        base_score: float,
        situation: str
    ) -> Optional[EnhancedAction]:
        """Helper to select best action from list."""
        scores = {}
        for action_type in action_types:
            action = create_action_from_type(action_type)
            score = score_action_for_person(self.person, action, base_score)
            scores[action_type] = score
        
        valid_scores = {a: s for a, s in scores.items() if s > -1e8}
        if not valid_scores:
            return None
        
        best_action_type = max(valid_scores, key=valid_scores.get)
        best_action = create_action_from_type(
            best_action_type,
            reason=f"Situation: {situation} (score={valid_scores[best_action_type]:.2f})"
        )
        
        self.affordance_system.execute_action(best_action)
        return best_action
    
    def check_for_loops(self) -> bool:
        """Check if stuck in action loop."""
        return self.affordance_system.detect_action_loop()
    
    def break_loop(self, being_state: any) -> Optional[EnhancedAction]:
        """
        Break out of action loop by selecting alternative action.
        
        Args:
            being_state: BeingState for context
        
        Returns:
            Alternative action or None
        """
        if not self.affordance_system.context.recent_actions:
            return None
        
        current = self.affordance_system.context.recent_actions[-1]
        alternatives = self.affordance_system.suggest_alternative_actions(current, count=3)
        
        if not alternatives:
            logger.warning("[ActionSelector] No alternatives to break loop")
            return None
        
        logger.info(f"[ActionSelector] Breaking loop, trying alternatives: {[a.value for a in alternatives]}")
        
        # Score alternatives
        scores = {}
        for action_type in alternatives:
            action = create_action_from_type(action_type)
            score = score_action_for_person(self.person, action, base_score=0.5)
            scores[action_type] = score
        
        valid_scores = {a: s for a, s in scores.items() if s > -1e8}
        if not valid_scores:
            return None
        
        best_action_type = max(valid_scores, key=valid_scores.get)
        best_action = create_action_from_type(
            best_action_type,
            reason=f"Breaking loop (score={valid_scores[best_action_type]:.2f})"
        )
        
        self.affordance_system.execute_action(best_action)
        return best_action
    
    def get_stats(self) -> Dict:
        """Get selector statistics."""
        affordance_stats = self.affordance_system.get_stats()
        
        return {
            'person': self.person.identity.name,
            'archetype': self.person.identity.archetype,
            **affordance_stats
        }


# ========================================
# Utility Functions
# ========================================

def score_enhanced_action(
    person: PersonModel,
    action: EnhancedAction,
    base_score: float = 0.5
) -> float:
    """
    Score an enhanced action using PersonModel.
    
    Wrapper around score_action_for_person that handles
    EnhancedAction-specific features.
    
    Args:
        person: PersonModel
        action: EnhancedAction to score
        base_score: Base score
    
    Returns:
        Final score
    """
    # Get affordance
    affordance = get_affordance(action.action_type)
    
    # Adjust base score by affordance priority
    if affordance:
        priority_bonus = (affordance.priority - 5) * 0.05  # -0.25 to +0.25
        adjusted_base = base_score + priority_bonus
    else:
        adjusted_base = base_score
    
    # Score with PersonModel
    score = score_action_for_person(person, action, adjusted_base)
    
    # Adjust by intensity
    score *= action.intensity
    
    return score


def create_action_sequence(
    action_types: List[EnhancedActionType],
    delays: Optional[List[float]] = None
) -> List[EnhancedAction]:
    """
    Create sequence of actions to execute in order.
    
    Args:
        action_types: List of action types
        delays: Optional delays between actions
    
    Returns:
        List of EnhancedActions
    """
    if delays is None:
        delays = [0.0] * len(action_types)
    
    actions = []
    for i, action_type in enumerate(action_types):
        action = create_action_from_type(
            action_type,
            reason=f"Sequence step {i+1}/{len(action_types)}"
        )
        if i < len(delays):
            action.delay = delays[i]
        actions.append(action)
    
    return actions


def get_action_description(action: EnhancedAction) -> str:
    """
    Get human-readable description of action.
    
    Args:
        action: EnhancedAction
    
    Returns:
        Description string
    """
    affordance = get_affordance(action.action_type)
    
    desc = action.action_type.value.replace('_', ' ').title()
    
    if action.target_id:
        desc += f" (target: {action.target_id})"
    
    if action.direction:
        desc += f" ({action.direction})"
    
    if action.intensity != 1.0:
        desc += f" (intensity: {action.intensity:.1f})"
    
    if affordance:
        desc += f" - {affordance.description}"
    
    return desc
