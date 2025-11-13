"""
Temporal Binding System

Ensures perception→action→outcome loops close properly, preventing
the perception-action decoupling problem identified in Skyrim debugging.

Key Innovation: Links perception at time T to action at time T+1,
creating genuine temporal coherence rather than just spatial integration.
"""

import time
import hashlib
from collections import deque
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class TemporalBinding:
    """Links perception to action across time."""
    perception_timestamp: float
    perception_content: Dict[str, Any]
    action_timestamp: float
    action_taken: str
    outcome: Optional[str] = None
    coherence_delta: float = 0.0
    success: bool = False
    binding_id: str = ""
    
    def __post_init__(self):
        """Generate unique binding ID."""
        if not self.binding_id:
            self.binding_id = hashlib.md5(
                f"{self.perception_timestamp}{self.action_taken}".encode()
            ).hexdigest()[:8]


class TemporalCoherenceTracker:
    """
    Ensures perception→action→outcome loops close properly.
    
    Solves the perception-action decoupling problem where the system
    sees but doesn't act on what it sees.
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize temporal coherence tracker.
        
        Args:
            window_size: Number of recent bindings to track
        """
        self.bindings: deque[TemporalBinding] = deque(maxlen=window_size)
        self.unclosed_loops = 0
        self.total_bindings = 0
        self.successful_loops = 0
        
        # Stuck detection
        self.visual_similarity_threshold = 0.95
        self.stuck_loop_count = 0
        
        logger.info("[TEMPORAL] Temporal coherence tracker initialized")
    
    def bind_perception_to_action(
        self,
        perception: Dict[str, Any],
        action: str
    ) -> str:
        """
        Create perception→action binding.
        
        Args:
            perception: Perception data at time T
            action: Action taken at time T+1
            
        Returns:
            Binding ID for later closure
        """
        binding = TemporalBinding(
            perception_timestamp=time.time(),
            perception_content=perception.copy(),
            action_timestamp=time.time(),
            action_taken=action
        )
        
        self.bindings.append(binding)
        self.unclosed_loops += 1
        self.total_bindings += 1
        
        # Check for stuck loops
        visual_sim = perception.get('visual_similarity', 0.0)
        if visual_sim > self.visual_similarity_threshold:
            self.stuck_loop_count += 1
            logger.warning(
                f"[TEMPORAL] Potential stuck loop detected: "
                f"visual_similarity={visual_sim:.3f}, "
                f"stuck_count={self.stuck_loop_count}"
            )
        else:
            self.stuck_loop_count = 0
        
        logger.debug(
            f"[TEMPORAL] Bound perception→action: {action} "
            f"(binding_id={binding.binding_id})"
        )
        
        return binding.binding_id
    
    def close_loop(
        self,
        binding_id: str,
        outcome: str,
        coherence_delta: float,
        success: bool = True
    ):
        """
        Close perception→action→outcome loop.
        
        Args:
            binding_id: ID from bind_perception_to_action
            outcome: Observed outcome
            coherence_delta: Change in coherence
            success: Whether the action was successful
        """
        for binding in self.bindings:
            if binding.binding_id == binding_id:
                binding.outcome = outcome
                binding.coherence_delta = coherence_delta
                binding.success = success
                self.unclosed_loops -= 1
                
                if success:
                    self.successful_loops += 1
                
                # Check for perception-action mismatch
                if coherence_delta < 0:
                    visual_sim = binding.perception_content.get('visual_similarity', 0)
                    logger.warning(
                        f"[TEMPORAL] Perception-action mismatch: "
                        f"visual_similarity={visual_sim:.3f}, "
                        f"action={binding.action_taken}, "
                        f"coherence_delta={coherence_delta:.3f}"
                    )
                
                logger.debug(
                    f"[TEMPORAL] Loop closed: {binding.action_taken} → {outcome} "
                    f"(Δ𝒞={coherence_delta:+.3f})"
                )
                break
    
    def get_unclosed_ratio(self) -> float:
        """
        Get ratio of unclosed loops.
        
        Returns:
            Ratio (0.0=all closed, 1.0=all open)
        """
        if not self.bindings:
            return 0.0
        return self.unclosed_loops / len(self.bindings)
    
    def get_success_rate(self) -> float:
        """Get success rate of closed loops."""
        if self.total_bindings == 0:
            return 0.0
        
        closed_loops = self.total_bindings - self.unclosed_loops
        if closed_loops == 0:
            return 0.0
        
        return self.successful_loops / closed_loops
    
    def is_stuck(self) -> bool:
        """Check if system appears stuck in a loop."""
        return self.stuck_loop_count >= 3
    
    def get_recent_bindings(self, count: int = 5) -> List[TemporalBinding]:
        """Get recent bindings for analysis."""
        return list(self.bindings)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get temporal coherence statistics."""
        return {
            'total_bindings': self.total_bindings,
            'unclosed_loops': self.unclosed_loops,
            'unclosed_ratio': self.get_unclosed_ratio(),
            'success_rate': self.get_success_rate(),
            'successful_loops': self.successful_loops,
            'stuck_loop_count': self.stuck_loop_count,
            'is_stuck': self.is_stuck(),
            'window_size': len(self.bindings),
        }
    
    def reset_stuck_counter(self):
        """Reset stuck loop counter after successful action."""
        if self.stuck_loop_count > 0:
            logger.info(
                f"[TEMPORAL] Stuck loop broken after {self.stuck_loop_count} cycles"
            )
        self.stuck_loop_count = 0
