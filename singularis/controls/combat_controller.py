"""CombatController: heuristic but effective combat AI."""

import random
from typing import Dict, Any
from .action_space import HighLevelAction


class CombatController:
    """Provides competent combat behavior using heuristics."""
    
    def __init__(self, low_health_threshold: float = 40.0, critical_health_threshold: float = 25.0):
        self.low_health = low_health_threshold
        self.critical_health = critical_health_threshold
        self.stats = {'decisions_made': 0, 'attacks': 0, 'blocks': 0, 'retreats': 0, 'tactical_moves': 0}
        self.recent_actions = []
    
    def choose_combat_action(self, game_state: Dict[str, Any]) -> HighLevelAction:
        self.stats['decisions_made'] += 1
        health = game_state.get('health', 100.0)
        stamina = game_state.get('stamina', 100.0)
        enemies = game_state.get('enemies_nearby', 1)
        
        if health < self.critical_health:
            self.stats['retreats'] += 1
            return HighLevelAction.RETREAT_FROM_TARGET
        
        if health < self.low_health:
            if random.random() < 0.6:
                self.stats['blocks'] += 1
                return HighLevelAction.BLOCK
            else:
                self.stats['retreats'] += 1
                return HighLevelAction.RETREAT_FROM_TARGET
        
        if enemies >= 3 and stamina > 50:
            self.stats['attacks'] += 1
            return HighLevelAction.POWER_ATTACK
        
        r = random.random()
        if r < 0.50:
            self.stats['attacks'] += 1
            return HighLevelAction.QUICK_ATTACK
        elif r < 0.70:
            if stamina > 50:
                self.stats['attacks'] += 1
                return HighLevelAction.POWER_ATTACK
            else:
                self.stats['attacks'] += 1
                return HighLevelAction.QUICK_ATTACK
        elif r < 0.85:
            self.stats['blocks'] += 1
            return HighLevelAction.BLOCK
        elif r < 0.92:
            self.stats['tactical_moves'] += 1
            return HighLevelAction.CIRCLE_LEFT if random.random() < 0.5 else HighLevelAction.CIRCLE_RIGHT
        else:
            self.stats['tactical_moves'] += 1
            return HighLevelAction.DODGE
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
