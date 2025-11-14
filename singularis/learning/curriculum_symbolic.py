"""Symbolic Logic Rules for Curriculum-Aware Learning."""

from typing import Dict, Any, List
from .curriculum_reward import CurriculumStage

class CurriculumSymbolicRules:
    """Symbolic rules that guide curriculum progression."""
    
    def __init__(self):
        self.rules = {
            CurriculumStage.STAGE_0_LOCOMOTION: self._locomotion_rules,
            CurriculumStage.STAGE_1_NAVIGATION: self._navigation_rules,
            CurriculumStage.STAGE_2_TARGET_ACQUISITION: self._target_rules,
            CurriculumStage.STAGE_3_DEFENSE: self._defense_rules,
            CurriculumStage.STAGE_4_COMBAT_1V1: self._combat_rules,
        }
    
    def get_rules(self, stage: CurriculumStage) -> List[str]:
        """Get symbolic rules for current curriculum stage."""
        handler = self.rules.get(stage)
        return handler() if handler else []
    
    def _locomotion_rules(self) -> List[str]:
        return [
            "IF idle THEN move_forward",
            "IF moved_recently AND visual_changed THEN continue_exploration",
            "IF stuck_counter > 3 THEN turn_large",
        ]
    
    def _navigation_rules(self) -> List[str]:
        return [
            "IF visual_similarity > 0.95 AND action == move_forward THEN turn_or_jump",
            "IF visual_similarity < 0.80 THEN reward_exploration",
            "IF stuck_counter > 5 THEN large_turn_and_jump",
        ]
    
    def _target_rules(self) -> List[str]:
        return [
            "IF enemy_visible THEN approach_target",
            "IF close_to_target THEN attack",
            "IF attack_missed THEN adjust_aim",
        ]
    
    def _defense_rules(self) -> List[str]:
        return [
            "IF health < 40 AND in_combat THEN block_or_retreat",
            "IF taking_damage THEN dodge",
            "IF health_stable AND in_combat THEN reward_defense",
        ]
    
    def _combat_rules(self) -> List[str]:
        return [
            "IF health > 60 AND enemy_nearby THEN attack",
            "IF health < 40 THEN defensive_stance",
            "IF enemy_defeated THEN large_reward",
            "IF taking_damage AND stamina > 50 THEN power_attack",
        ]
    
    def evaluate_rule(self, rule: str, state: Dict[str, Any]) -> bool:
        """Simple rule evaluation (can be enhanced with proper parser)."""
        rule_lower = rule.lower()
        
        if "health <" in rule_lower:
            threshold = float(rule_lower.split("health <")[1].split()[0])
            return state.get('health', 100) < threshold
        
        if "visual_similarity >" in rule_lower:
            threshold = float(rule_lower.split("visual_similarity >")[1].split()[0])
            return state.get('visual_similarity', 0) > threshold
        
        if "in_combat" in rule_lower:
            return state.get('in_combat', False)
        
        if "enemy_nearby" in rule_lower or "enemy_visible" in rule_lower:
            return state.get('enemies_nearby', 0) > 0
        
        return False
    
    def get_action_from_rule(self, rule: str) -> str:
        """Extract action from THEN clause."""
        if "THEN" in rule:
            action_part = rule.split("THEN")[1].strip()
            return action_part.split()[0]
        return "move_forward"
