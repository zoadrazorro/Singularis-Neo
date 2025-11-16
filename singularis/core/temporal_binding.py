"""
Temporal Binding System

Ensures perception→action→outcome loops close properly, preventing
the perception-action decoupling problem identified in Skyrim debugging.

Key Innovation: Links perception at time T to action at time T+1,
creating genuine temporal coherence rather than just spatial integration.
"""

import time
import asyncio
import hashlib
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
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

    # Multi-modal visual analysis
    gemini_visual: Optional[str] = None
    hyperbolic_visual: Optional[str] = None
    video_interpretation: Optional[str] = None
    gpt_video_analysis: Optional[str] = None
    bdh_sigma_snapshots: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate unique binding ID."""
        if not self.binding_id:
            self.binding_id = hashlib.md5(
                f"{self.perception_timestamp}{self.action_taken}".encode()
            ).hexdigest()[:8]


class TemporalCoherenceTracker:
    """
    Tracks and manages the temporal binding of perception, action, and outcome.

    This class is crucial for ensuring that the AGI's actions are coherently
    linked to its perceptions and their resulting outcomes. It helps solve the
    perception-action decoupling problem by tracking loops and detecting when
    the system is stuck.
    """
    
    def __init__(self, window_size: int = 20, unclosed_timeout: float = 30.0):
        """
        Initializes the TemporalCoherenceTracker.

        Args:
            window_size (int, optional): The number of recent bindings to track.
                                         Defaults to 20.
            unclosed_timeout (float, optional): The time in seconds before an unclosed
                                                binding is considered stale and
                                                automatically closed. Defaults to 30.0.
        """
        self.bindings: deque[TemporalBinding] = deque(maxlen=window_size)
        self.unclosed_loops = 0
        self.total_bindings = 0
        self.successful_loops = 0
        self.unclosed_timeout = unclosed_timeout
        
        # Track individual unclosed bindings (CRITICAL: prevents memory leak)
        self.unclosed_bindings: Dict[str, float] = {}  # binding_id → timestamp
        
        # Stuck detection
        self.visual_similarity_threshold = 0.95
        self.stuck_loop_count = 0
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"[TEMPORAL] Temporal coherence tracker initialized "
            f"(window={window_size}, timeout={unclosed_timeout}s)"
        )
    
    def bind_perception_to_action(
        self,
        perception: Dict[str, Any],
        action: str,
        gemini_visual: Optional[str] = None,
        hyperbolic_visual: Optional[str] = None,
        video_interpretation: Optional[str] = None,
        gpt_video_analysis: Optional[str] = None,
        bdh_sigma_snapshots: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Creates a temporal binding between a perception and a subsequent action.

        This method records the perception, the action taken, and various
        multi-modal visual analyses. It also checks for potential stuck loops
        based on visual similarity.

        Args:
            perception (Dict[str, Any]): The perception data at the time of the action.
            action (str): The action that was taken.
            gemini_visual (Optional[str], optional): Gemini visual analysis. Defaults to None.
            hyperbolic_visual (Optional[str], optional): Hyperbolic Nemotron visual analysis. Defaults to None.
            video_interpretation (Optional[str], optional): Streaming video interpreter output. Defaults to None.
            gpt_video_analysis (Optional[str], optional): GPT-4 video analysis. Defaults to None.
            bdh_sigma_snapshots (Optional[Dict[str, Any]], optional): BDH sigma snapshots. Defaults to None.

        Returns:
            str: The unique ID of the created binding, which can be used to close the loop later.
        """
        binding = TemporalBinding(
            perception_timestamp=time.time(),
            perception_content=perception.copy(),
            action_timestamp=time.time(),
            action_taken=action,
            gemini_visual=gemini_visual,
            hyperbolic_visual=hyperbolic_visual,
            video_interpretation=video_interpretation,
            gpt_video_analysis=gpt_video_analysis,
            bdh_sigma_snapshots=bdh_sigma_snapshots or {}
        )
        
        self.bindings.append(binding)
        self.unclosed_loops += 1
        self.total_bindings += 1
        
        # Track for timeout (CRITICAL: prevents unbounded growth)
        self.unclosed_bindings[binding.binding_id] = time.time()
        
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
            f"(binding_id={binding.binding_id}, unclosed={len(self.unclosed_bindings)})"
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
        Closes a perception-action-outcome loop.

        This method updates the specified binding with the outcome of the action,
        the change in coherence, and whether the action was successful. It also
        checks for perception-action mismatches based on the coherence delta.

        Args:
            binding_id (str): The ID of the binding to close, as returned by `bind_perception_to_action`.
            outcome (str): The observed outcome of the action.
            coherence_delta (float): The change in coherence resulting from the action.
            success (bool, optional): Whether the action was successful. Defaults to True.
        """
        for binding in self.bindings:
            if binding.binding_id == binding_id:
                binding.outcome = outcome
                binding.coherence_delta = coherence_delta
                binding.success = success
                self.unclosed_loops -= 1
                
                # Remove from unclosed tracking (CRITICAL: prevents memory leak)
                if binding_id in self.unclosed_bindings:
                    del self.unclosed_bindings[binding_id]
                
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
                    f"(Δ𝒞={coherence_delta:+.3f}, unclosed={len(self.unclosed_bindings)})"
                )
                break
    
    def get_unclosed_ratio(self) -> float:
        """
        Gets the ratio of unclosed loops to the total number of bindings.

        Returns:
            float: A ratio between 0.0 (all loops closed) and 1.0 (all loops open).
        """
        if not self.bindings:
            return 0.0
        return self.unclosed_loops / len(self.bindings)
    
    def get_success_rate(self) -> float:
        """
        Gets the success rate of the closed loops.

        Returns:
            float: The success rate, between 0.0 and 1.0.
        """
        if self.total_bindings == 0:
            return 0.0
        
        closed_loops = self.total_bindings - self.unclosed_loops
        if closed_loops == 0:
            return 0.0
        
        return self.successful_loops / closed_loops
    
    def is_stuck(self) -> bool:
        """
        Checks if the system appears to be stuck in a loop.

        Returns:
            bool: True if the stuck loop count is 3 or more, False otherwise.
        """
        return self.stuck_loop_count >= 3
    
    def get_recent_bindings(self, count: int = 5) -> List[TemporalBinding]:
        """
        Gets a list of the most recent temporal bindings.

        Args:
            count (int, optional): The number of recent bindings to retrieve.
                                 Defaults to 5.

        Returns:
            List[TemporalBinding]: A list of the most recent bindings.
        """
        return list(self.bindings)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the temporal coherence tracker.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        return {
            'total_bindings': self.total_bindings,
            'unclosed_loops': self.unclosed_loops,
            'unclosed_ratio': self.get_unclosed_ratio(),
            'success_rate': self.get_success_rate(),
            'successful_loops': self.successful_loops,
            'stuck_loop_count': self.stuck_loop_count,
            'is_stuck': self.is_stuck(),
            'window_size': len(self.bindings),
            'bdh_snapshot_count': sum(1 for binding in self.bindings if binding.bdh_sigma_snapshots),
        }
    
    def reset_stuck_counter(self):
        """Resets the stuck loop counter, typically after a successful action."""
        if self.stuck_loop_count > 0:
            logger.info(
                f"[TEMPORAL] Stuck loop broken after {self.stuck_loop_count} cycles"
            )
        self.stuck_loop_count = 0
    
    async def start(self):
        """Starts the background task that cleans up stale bindings."""
        if self._cleanup_task is None:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_bindings())
            logger.info("[TEMPORAL] Cleanup task started")
    
    async def _cleanup_stale_bindings(self):
        """Periodically clean up stale unclosed bindings."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                now = time.time()
                stale = []
                
                for binding_id, timestamp in list(self.unclosed_bindings.items()):
                    if now - timestamp > self.unclosed_timeout:
                        stale.append(binding_id)
                
                if stale:
                    logger.warning(
                        f"[TEMPORAL] Auto-closing {len(stale)} stale bindings "
                        f"(timeout: {self.unclosed_timeout}s)"
                    )
                    
                    for binding_id in stale:
                        # Force-close with failure outcome
                        self.close_loop(
                            binding_id=binding_id,
                            outcome="timeout_failure",
                            coherence_delta=-0.2,
                            success=False
                        )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[TEMPORAL] Cleanup error: {e}")
    
    async def close(self):
        """Shuts down the temporal tracker and cleans up the background task."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("[TEMPORAL] Temporal tracker closed")
