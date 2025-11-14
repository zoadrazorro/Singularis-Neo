"""Navigator: intelligent exploration that doesn't just spam actions."""

import random
from typing import Dict, Any, Optional
from .action_space import HighLevelAction


class Navigator:
    """
    Handles exploration movement with basic stuck detection.
    
    Provides sensible exploration behavior:
    - Walk forward most of the time
    - Occasionally turn to scan environment
    - Detect stuck loops via visual similarity
    - Escape stuck states with large turns or jumps
    """
    
    def __init__(self, stuck_threshold: int = 6):
        """
        Initialize navigator.
        
        Args:
            stuck_threshold: Cycles of similarity before considering stuck
        """
        self.stuck_threshold = stuck_threshold
        self.last_visual_hash = None
        self.stuck_counter = 0
        self.last_action = None
        self.action_repetitions = 0
        self.camera_action_count = 0  # Track camera-only actions
        
        self.stats = {
            'suggestions_made': 0,
            'stuck_detections': 0,
            'recovery_actions': 0,
            'camera_stuck_detections': 0,
        }
    
    def _hash_visual(self, perception: Dict[str, Any]) -> int:
        """Quick hash of visual state for stuck detection."""
        # Use visual embedding if available
        embedding = perception.get('visual_embedding')
        if embedding is not None:
            try:
                import numpy as np
                return hash(tuple(embedding[:10]))  # First 10 dimensions
            except:
                pass
        
        # Fallback: hash text summary
        text = str(perception.get('gemini_analysis', ''))[:300]
        location = perception.get('location', '')
        return hash(text + location)
    
    def suggest_exploration_action(
        self,
        game_state: Dict[str, Any],
        perception: Optional[Dict[str, Any]] = None
    ) -> HighLevelAction:
        """
        Suggest next exploration action.
        
        Args:
            game_state: Current game state
            perception: Optional perception data for stuck detection
            
        Returns:
            Suggested exploration action
        """
        self.stats['suggestions_made'] += 1
        
        # Check for stuck loops via visual similarity
        if perception is not None:
            visual_hash = self._hash_visual(perception)
            
            if self.last_visual_hash is not None and visual_hash == self.last_visual_hash:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
            
            self.last_visual_hash = visual_hash
        
        # STUCK DETECTED: force recovery
        if self.stuck_counter >= self.stuck_threshold:
            self.stats['stuck_detections'] += 1
            self.stats['recovery_actions'] += 1
            self.stuck_counter = 0  # Reset after recovery attempt
            
            print(f"[NAVIGATOR] Stuck detected! Forcing recovery action")
            
            # Escalating recovery actions
            r = random.random()
            if r < 0.4:
                return HighLevelAction.TURN_LEFT_LARGE
            elif r < 0.8:
                return HighLevelAction.TURN_RIGHT_LARGE
            else:
                return HighLevelAction.JUMP
        
        # NORMAL EXPLORATION
        # Weight actions to prefer forward progress
        r = random.random()
        
        if r < 0.50:
            # Forward movement (50%)
            action = HighLevelAction.STEP_FORWARD
        elif r < 0.65:
            # Look around (15%)
            action = HighLevelAction.LOOK_AROUND
        elif r < 0.75:
            # Small turn left (10%)
            action = HighLevelAction.TURN_LEFT_SMALL
        elif r < 0.85:
            # Small turn right (10%)
            action = HighLevelAction.TURN_RIGHT_SMALL
        elif r < 0.90:
            # Jump for terrain (5%)
            action = HighLevelAction.JUMP
        elif r < 0.95:
            # Large turn (5%)
            if random.random() < 0.5:
                action = HighLevelAction.TURN_LEFT_LARGE
            else:
                action = HighLevelAction.TURN_RIGHT_LARGE
        else:
            # Activate to interact (5%)
            action = HighLevelAction.ACTIVATE
        
        # Track action repetition
        if action == self.last_action:
            self.action_repetitions += 1
        else:
            self.action_repetitions = 0
        
        self.last_action = action
        
        # If repeating same action too much, force variety
        if self.action_repetitions > 4:
            print(f"[NAVIGATOR] Breaking repetition of {action.name}")
            self.action_repetitions = 0
            # Do something different
            if action == HighLevelAction.STEP_FORWARD:
                return HighLevelAction.TURN_LEFT_SMALL
            else:
                return HighLevelAction.STEP_FORWARD
        
        return action
    
    def reset(self):
        """Reset navigator state (e.g., when entering new area)."""
        self.last_visual_hash = None
        self.stuck_counter = 0
        self.last_action = None
        self.action_repetitions = 0
        self.camera_action_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get navigator statistics."""
        return {
            **self.stats,
            'stuck_counter': self.stuck_counter,
            'action_repetitions': self.action_repetitions,
        }
