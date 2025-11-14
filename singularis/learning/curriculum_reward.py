"""Curriculum-Aware RL Reward: blends coherence + game progress."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class CurriculumStage(Enum):
    STAGE_0_LOCOMOTION = 0
    STAGE_1_NAVIGATION = 1
    STAGE_2_TARGET_ACQUISITION = 2
    STAGE_3_DEFENSE = 3
    STAGE_4_COMBAT_1V1 = 4
    STAGE_5_MASTERY = 5

@dataclass
class CurriculumProgress:
    current_stage: CurriculumStage
    stage_cycles: int
    stage_successes: int
    stage_failures: int
    advancement_threshold: int = 20
    regression_threshold: int = 30

class CurriculumRewardFunction:
    def __init__(self, coherence_weight=0.6, progress_weight=0.4):
        self.coherence_weight = coherence_weight
        self.progress_weight = progress_weight
        self.progress = CurriculumProgress(CurriculumStage.STAGE_0_LOCOMOTION, 0, 0, 0)
    
    def compute_reward(self, state_before, action, state_after, consciousness_before=None, consciousness_after=None):
        coherence_reward = 0.0
        if consciousness_before and consciousness_after:
            delta_c = consciousness_after.coherence - consciousness_before.coherence
            coherence_reward = delta_c * 3.0
        
        progress_reward = self._stage_reward(state_before, action, state_after)
        total = self.coherence_weight * coherence_reward + self.progress_weight * progress_reward
        
        if progress_reward > 1.0:
            self.progress.stage_successes += 1
        elif progress_reward < -1.0:
            self.progress.stage_failures += 1
        
        if self.progress.stage_successes >= self.progress.advancement_threshold:
            self._advance_stage()
            total += 5.0
        
        return total
    
    def _stage_reward(self, sb, action, sa):
        stage = self.progress.current_stage
        if stage == CurriculumStage.STAGE_0_LOCOMOTION:
            return 1.0 if any(x in action.lower() for x in ['forward','turn','look']) else 0.0
        elif stage == CurriculumStage.STAGE_1_NAVIGATION:
            vsim = sa.get('visual_similarity', 0)
            return 2.0 if vsim < 0.8 else -2.0 if vsim > 0.95 else 0.0
        elif stage == CurriculumStage.STAGE_2_TARGET_ACQUISITION:
            return 2.0 if 'attack' in action.lower() else 0.5 if 'approach' in action.lower() else 0.0
        elif stage == CurriculumStage.STAGE_3_DEFENSE:
            hdelta = sa.get('health',100) - sb.get('health',100)
            return 2.0 if hdelta >= 0 and sa.get('in_combat') else hdelta * 0.5
        elif stage == CurriculumStage.STAGE_4_COMBAT_1V1:
            hdelta = sa.get('health',100) - sb.get('health',100)
            enemies_before = sb.get('enemies_nearby',0)
            enemies_after = sa.get('enemies_nearby',0)
            r = 0.0
            if enemies_after < enemies_before:
                r += 5.0
            if 'attack' in action.lower():
                r += 1.0
            if hdelta < 0:
                r += hdelta * 0.3
            return r
        return 0.0
    
    def _advance_stage(self):
        stages = list(CurriculumStage)
        idx = stages.index(self.progress.current_stage)
        if idx < len(stages) - 1:
            self.progress.current_stage = stages[idx + 1]
            self.progress.stage_cycles = 0
            self.progress.stage_successes = 0
            self.progress.stage_failures = 0
            print(f"[CURRICULUM] Advanced to {self.progress.current_stage.name}")
    
    def get_current_stage(self):
        return self.progress.current_stage
