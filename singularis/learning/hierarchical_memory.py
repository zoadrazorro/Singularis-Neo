"""
Hierarchical Memory Consolidation

Implements two-tier memory: episodic → semantic

Episodic memories are consolidated into semantic knowledge patterns,
enabling genuine learning rather than just experience storage.
"""

import time
import asyncio
from collections import deque, Counter
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger
import numpy as np


@dataclass
class EpisodicMemory:
    """A single episodic memory."""
    timestamp: float
    scene_type: str
    action: str
    outcome: str
    outcome_success: bool
    coherence_delta: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticPattern:
    """A consolidated semantic pattern."""
    pattern_key: str
    scene_type: str
    optimal_action: str
    success_rate: float
    sample_size: int
    confidence: float
    first_learned: float
    last_updated: float
    contexts: List[Dict[str, Any]] = field(default_factory=list)


class HierarchicalMemory:
    """
    Two-tier memory system: episodic → semantic.
    
    Episodic tier: Recent experiences (short-term)
    Semantic tier: Consolidated patterns (long-term)
    """
    
    def __init__(
        self,
        episodic_capacity: int = 1000,
        consolidation_threshold: int = 10,
        min_pattern_samples: int = 3,
        consolidation_interval: float = 60.0
    ):
        """
        Initialize hierarchical memory.
        
        Args:
            episodic_capacity: Max episodic memories to store
            consolidation_threshold: Episodes needed to trigger consolidation
            min_pattern_samples: Min samples to form semantic pattern
            consolidation_interval: Min seconds between consolidations
        """
        self.episodic: deque[EpisodicMemory] = deque(maxlen=episodic_capacity)
        self.semantic: Dict[str, SemanticPattern] = {}
        
        self.consolidation_threshold = consolidation_threshold
        self.min_pattern_samples = min_pattern_samples
        self.consolidation_interval = consolidation_interval
        
        self.last_consolidation_time = 0.0
        self.total_consolidations = 0
        self.patterns_learned = 0
        
        logger.info(
            f"[MEMORY] Hierarchical memory initialized "
            f"(episodic={episodic_capacity}, threshold={consolidation_threshold})"
        )
    
    def store_episode(
        self,
        scene_type: str,
        action: str,
        outcome: str,
        outcome_success: bool,
        coherence_delta: float,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store episodic memory.
        
        Args:
            scene_type: Type of scene (combat, exploration, etc.)
            action: Action taken
            outcome: Observed outcome
            outcome_success: Whether action was successful
            coherence_delta: Change in coherence
            context: Additional context
            metadata: Additional metadata
        """
        episode = EpisodicMemory(
            timestamp=time.time(),
            scene_type=scene_type,
            action=action,
            outcome=outcome,
            outcome_success=outcome_success,
            coherence_delta=coherence_delta,
            context=context or {},
            metadata=metadata or {}
        )
        
        self.episodic.append(episode)
        
        logger.debug(
            f"[MEMORY] Stored episode: {scene_type}/{action} → {outcome} "
            f"(success={outcome_success}, Δ𝒞={coherence_delta:+.3f})"
        )
        
        # Check for consolidation opportunity
        if self._should_consolidate():
            asyncio.create_task(self._consolidate())
    
    def _should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        # Need enough episodes
        if len(self.episodic) < self.consolidation_threshold:
            return False
        
        # Don't consolidate too frequently
        time_since_last = time.time() - self.last_consolidation_time
        if time_since_last < self.consolidation_interval:
            return False
        
        return True
    
    async def _consolidate(self):
        """
        Consolidate episodic → semantic knowledge.
        
        Finds patterns in recent episodes and creates/updates
        semantic knowledge.
        """
        self.last_consolidation_time = time.time()
        self.total_consolidations += 1
        
        logger.info(
            f"[MEMORY] Starting consolidation #{self.total_consolidations} "
            f"({len(self.episodic)} episodes)"
        )
        
        # Group episodes by scene type
        by_scene: Dict[str, List[EpisodicMemory]] = {}
        
        recent_episodes = list(self.episodic)[-self.consolidation_threshold:]
        
        for episode in recent_episodes:
            scene = episode.scene_type
            if scene not in by_scene:
                by_scene[scene] = []
            by_scene[scene].append(episode)
        
        # Extract patterns for each scene
        patterns_formed = 0
        
        for scene, episodes in by_scene.items():
            if len(episodes) < self.min_pattern_samples:
                continue
            
            # Find successful actions
            successful = [e for e in episodes if e.outcome_success]
            
            if not successful:
                continue
            
            # Count action frequencies
            actions = [e.action for e in successful]
            action_counts = Counter(actions)
            
            # Get most common successful action
            most_common_action, count = action_counts.most_common(1)[0]
            
            # Compute success rate
            total_attempts = sum(1 for e in episodes if e.action == most_common_action)
            success_rate = count / total_attempts if total_attempts > 0 else 0.0
            
            # Only store if success rate is high enough
            if success_rate < 0.5:
                continue
            
            # Compute confidence (more samples = higher confidence)
            confidence = self._compute_confidence(count, len(episodes))
            
            # Create or update semantic pattern
            pattern_key = f"{scene}_optimal_action"
            
            # Collect contexts from successful episodes
            successful_contexts = [
                e.context for e in successful
                if e.action == most_common_action
            ]
            
            if pattern_key in self.semantic:
                # Update existing pattern
                pattern = self.semantic[pattern_key]
                pattern.success_rate = (
                    pattern.success_rate * 0.7 + success_rate * 0.3
                )  # Exponential smoothing
                pattern.sample_size += len(episodes)
                pattern.confidence = confidence
                pattern.last_updated = time.time()
                pattern.contexts.extend(successful_contexts)
                
                # Keep only recent contexts
                if len(pattern.contexts) > 20:
                    pattern.contexts = pattern.contexts[-20:]
                
                logger.info(
                    f"[MEMORY] Updated pattern: {pattern_key} → {most_common_action} "
                    f"(success_rate={pattern.success_rate:.2%}, n={pattern.sample_size})"
                )
            else:
                # Create new pattern
                pattern = SemanticPattern(
                    pattern_key=pattern_key,
                    scene_type=scene,
                    optimal_action=most_common_action,
                    success_rate=success_rate,
                    sample_size=len(episodes),
                    confidence=confidence,
                    first_learned=time.time(),
                    last_updated=time.time(),
                    contexts=successful_contexts[:20]  # Keep max 20
                )
                
                self.semantic[pattern_key] = pattern
                self.patterns_learned += 1
                patterns_formed += 1
                
                logger.info(
                    f"[MEMORY] Learned new pattern: {pattern_key} → {most_common_action} "
                    f"({count}/{len(episodes)} success, confidence={confidence:.2%})"
                )
        
        logger.info(
            f"[MEMORY] Consolidation complete: "
            f"{patterns_formed} new patterns, "
            f"{len(self.semantic)} total patterns"
        )
    
    def _compute_confidence(self, successes: int, total: int) -> float:
        """
        Compute confidence score using Wilson score interval.
        
        Args:
            successes: Number of successes
            total: Total attempts
            
        Returns:
            Confidence score (0-1)
        """
        if total == 0:
            return 0.0
        
        # Wilson score with 95% confidence
        z = 1.96  # 95% confidence
        p = successes / total
        
        denominator = 1 + z**2 / total
        centre_adjusted = p + z**2 / (2 * total)
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        
        # Lower bound of confidence interval
        confidence = (centre_adjusted - margin) / denominator
        
        return max(0.0, min(1.0, confidence))
    
    def retrieve_semantic(
        self,
        scene_type: str,
        min_confidence: float = 0.3
    ) -> Optional[SemanticPattern]:
        """
        Retrieve consolidated semantic knowledge.
        
        Args:
            scene_type: Type of scene
            min_confidence: Minimum confidence threshold
            
        Returns:
            Semantic pattern or None if not found/low confidence
        """
        pattern_key = f"{scene_type}_optimal_action"
        pattern = self.semantic.get(pattern_key)
        
        if pattern and pattern.confidence >= min_confidence:
            logger.debug(
                f"[MEMORY] Retrieved pattern: {pattern_key} → {pattern.optimal_action} "
                f"(confidence={pattern.confidence:.2%})"
            )
            return pattern
        
        return None
    
    def get_all_patterns(
        self,
        min_confidence: float = 0.0
    ) -> List[SemanticPattern]:
        """Get all semantic patterns above confidence threshold."""
        return [
            p for p in self.semantic.values()
            if p.confidence >= min_confidence
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'episodic_count': len(self.episodic),
            'episodic_capacity': self.episodic.maxlen,
            'semantic_patterns': len(self.semantic),
            'total_consolidations': self.total_consolidations,
            'patterns_learned': self.patterns_learned,
            'avg_pattern_confidence': (
                np.mean([p.confidence for p in self.semantic.values()])
                if self.semantic else 0.0
            ),
            'avg_success_rate': (
                np.mean([p.success_rate for p in self.semantic.values()])
                if self.semantic else 0.0
            ),
        }
    
    def clear_episodic(self):
        """Clear episodic memory (keep semantic)."""
        self.episodic.clear()
        logger.info("[MEMORY] Cleared episodic memory")
    
    def get_recent_episodes(
        self,
        count: int = 10,
        scene_type: Optional[str] = None
    ) -> List[EpisodicMemory]:
        """Get recent episodic memories."""
        episodes = list(self.episodic)[-count:]
        
        if scene_type:
            episodes = [e for e in episodes if e.scene_type == scene_type]
        
        return episodes
