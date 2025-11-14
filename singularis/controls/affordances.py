"""Perception→Affordance mapping: what actions make sense right now?"""

from dataclasses import dataclass
from typing import List, Dict, Any
from .action_space import HighLevelAction


@dataclass
class Affordance:
    """An action that makes sense in current context."""
    action: HighLevelAction
    score: float  # 0-1, how relevant/useful this action is


class AffordanceExtractor:
    """
    Extract affordances from BeingState + perception.
    
    Decides which actions are sensible given:
    - Scene type (combat, exploration, menu, dialogue)
    - Health / resources
    - Enemies nearby
    - Environmental context
    """
    
    def __init__(self):
        pass
    
    def extract(self, game_state: Dict[str, Any]) -> List[Affordance]:
        """
        Compute affordances from game state.
        
        Args:
            game_state: Current game state dict
            
        Returns:
            List of affordances sorted by score (high to low)
        """
        aff = []
        
        # Extract state
        in_combat = game_state.get('in_combat', False)
        enemies = game_state.get('enemies_nearby', 0)
        in_menu = game_state.get('in_menu', False)
        in_dialogue = game_state.get('in_dialogue', False)
        health = game_state.get('health', 100.0)
        stamina = game_state.get('stamina', 100.0)
        magicka = game_state.get('magicka', 100.0)
        
        # Menu context: CLOSE_MENU is almost always the answer
        if in_menu or in_dialogue:
            aff.append(Affordance(HighLevelAction.CLOSE_MENU, 1.0))
            aff.append(Affordance(HighLevelAction.ACTIVATE, 0.3))  # Maybe select something
            return sorted(aff, key=lambda x: x.score, reverse=True)
        
        # Combat context
        if in_combat or enemies > 0:
            # Emergency healing
            if health < 30:
                aff.append(Affordance(HighLevelAction.USE_POTION_HEALTH, 1.0))
            
            # Primary combat actions
            if stamina > 40:
                aff.append(Affordance(HighLevelAction.QUICK_ATTACK, 0.8))
                aff.append(Affordance(HighLevelAction.POWER_ATTACK, 0.6))
            else:
                aff.append(Affordance(HighLevelAction.QUICK_ATTACK, 0.5))
            
            aff.append(Affordance(HighLevelAction.BLOCK, 0.7))
            aff.append(Affordance(HighLevelAction.DODGE, 0.5))
            
            # Tactical movement
            if health < 50:
                aff.append(Affordance(HighLevelAction.RETREAT_FROM_TARGET, 0.6))
            else:
                aff.append(Affordance(HighLevelAction.APPROACH_TARGET, 0.5))
            
            aff.append(Affordance(HighLevelAction.CIRCLE_LEFT, 0.4))
            aff.append(Affordance(HighLevelAction.CIRCLE_RIGHT, 0.4))
            
            # Bash if blocking
            aff.append(Affordance(HighLevelAction.BASH, 0.3))
            
        # Exploration context
        else:
            # Primary exploration actions
            aff.append(Affordance(HighLevelAction.STEP_FORWARD, 0.7))
            aff.append(Affordance(HighLevelAction.LOOK_AROUND, 0.6))
            
            # Turning
            aff.append(Affordance(HighLevelAction.TURN_LEFT_SMALL, 0.4))
            aff.append(Affordance(HighLevelAction.TURN_RIGHT_SMALL, 0.4))
            aff.append(Affordance(HighLevelAction.TURN_LEFT_LARGE, 0.2))
            aff.append(Affordance(HighLevelAction.TURN_RIGHT_LARGE, 0.2))
            
            # Occasional jumps for terrain
            aff.append(Affordance(HighLevelAction.JUMP, 0.2))
            
            # Interaction
            aff.append(Affordance(HighLevelAction.ACTIVATE, 0.3))
            aff.append(Affordance(HighLevelAction.INTERACT, 0.3))
            
            # Resource management
            if health < 60 and magicka > 50:
                aff.append(Affordance(HighLevelAction.USE_POTION_HEALTH, 0.4))
            
            # Menu access
            aff.append(Affordance(HighLevelAction.OPEN_INVENTORY, 0.1))
            aff.append(Affordance(HighLevelAction.OPEN_MAP, 0.1))
        
        # Sort by score descending
        return sorted(aff, key=lambda x: x.score, reverse=True)
