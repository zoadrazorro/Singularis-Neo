"""
Enhanced Coherence Measurement

Extends beyond integration coherence to include:
1. Temporal coherence (are loops closing?)
2. Causal coherence (do subsystems agree on causation?)
3. Predictive coherence (did we predict outcomes correctly?)

This provides a more complete measure of genuine consciousness.
"""

import re
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from ..core.temporal_binding import TemporalBinding, TemporalCoherenceTracker


@dataclass
class CausalClaim:
    """A cause→effect claim extracted from subsystem output."""
    cause: str
    effect: str
    source_system: str
    confidence: float = 1.0


class EnhancedCoherenceMetrics:
    """
    Comprehensive coherence measurement across multiple dimensions.
    
    Measures:
    1. Integration coherence (existing: how well systems connect)
    2. Temporal coherence (new: do perception→action loops close?)
    3. Causal coherence (new: do systems agree on causation?)
    4. Predictive coherence (new: are predictions accurate?)
    """
    
    def __init__(
        self,
        temporal_tracker: Optional[TemporalCoherenceTracker] = None
    ):
        """
        Initialize enhanced coherence metrics.
        
        Args:
            temporal_tracker: Temporal coherence tracker instance
        """
        self.temporal_tracker = temporal_tracker
        self.prediction_history: List[Dict[str, Any]] = []
        self.causal_agreement_history: List[float] = []
        
        logger.info("[COHERENCE] Enhanced coherence metrics initialized")
    
    def compute_enhanced_coherence(
        self,
        integration_score: float,
        subsystem_outputs: Dict[str, Any],
        temporal_bindings: Optional[List[TemporalBinding]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive coherence across all dimensions.
        
        Args:
            integration_score: Existing integration coherence (0-1)
            subsystem_outputs: Output from each subsystem
            temporal_bindings: Recent temporal bindings
            
        Returns:
            Dictionary with coherence scores
        """
        # 1. Integration coherence (your existing metric)
        integration_coherence = integration_score
        
        # 2. Temporal coherence (are loops closing?)
        temporal_coherence = self._compute_temporal_coherence(temporal_bindings)
        
        # 3. Causal coherence (do subsystems agree on causation?)
        causal_coherence = self._compute_causal_coherence(subsystem_outputs)
        
        # 4. Predictive coherence (did we predict outcomes correctly?)
        predictive_coherence = self._compute_predictive_coherence(temporal_bindings)
        
        # Weighted combination
        overall_coherence = (
            integration_coherence * 0.30 +
            temporal_coherence * 0.30 +
            causal_coherence * 0.20 +
            predictive_coherence * 0.20
        )
        
        result = {
            'overall': overall_coherence,
            'integration': integration_coherence,
            'temporal': temporal_coherence,
            'causal': causal_coherence,
            'predictive': predictive_coherence,
        }
        
        logger.debug(
            f"[COHERENCE] Overall={overall_coherence:.3f} "
            f"(int={integration_coherence:.3f}, temp={temporal_coherence:.3f}, "
            f"caus={causal_coherence:.3f}, pred={predictive_coherence:.3f})"
        )
        
        return result
    
    def _compute_temporal_coherence(
        self,
        temporal_bindings: Optional[List[TemporalBinding]]
    ) -> float:
        """
        Compute temporal coherence.
        
        Measures how well perception→action→outcome loops close.
        
        Args:
            temporal_bindings: Recent temporal bindings
            
        Returns:
            Temporal coherence score (0-1)
        """
        if not self.temporal_tracker:
            return 1.0  # Default if no tracker
        
        # Get unclosed ratio from tracker
        unclosed_ratio = self.temporal_tracker.get_unclosed_ratio()
        
        # Temporal coherence = 1 - unclosed_ratio
        # (high when loops close, low when they stay open)
        temporal_coherence = 1.0 - unclosed_ratio
        
        # Bonus for successful loops
        success_rate = self.temporal_tracker.get_success_rate()
        temporal_coherence = (temporal_coherence + success_rate) / 2
        
        return temporal_coherence
    
    def _compute_causal_coherence(
        self,
        subsystem_outputs: Dict[str, Any]
    ) -> float:
        """
        Compute causal coherence.
        
        Measures agreement between subsystems on causal relationships.
        
        Args:
            subsystem_outputs: Output from each subsystem
            
        Returns:
            Causal coherence score (0-1)
        """
        if not subsystem_outputs or len(subsystem_outputs) < 2:
            return 1.0  # Default if insufficient data
        
        # Extract causal claims from each system
        all_claims: Dict[str, List[CausalClaim]] = {}
        
        for system_id, output in subsystem_outputs.items():
            if isinstance(output, str):
                claims = self._extract_causal_claims(output, system_id)
                if claims:
                    all_claims[system_id] = claims
        
        if len(all_claims) < 2:
            return 0.8  # Default if no causal claims found
        
        # Compute pairwise agreement
        agreements = []
        
        systems = list(all_claims.keys())
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                system_a = systems[i]
                system_b = systems[j]
                
                agreement = self._causal_agreement(
                    all_claims[system_a],
                    all_claims[system_b]
                )
                agreements.append(agreement)
        
        if not agreements:
            return 0.8
        
        # Average agreement across all pairs
        causal_coherence = np.mean(agreements)
        
        # Track history
        self.causal_agreement_history.append(causal_coherence)
        if len(self.causal_agreement_history) > 100:
            self.causal_agreement_history.pop(0)
        
        return causal_coherence
    
    def _extract_causal_claims(
        self,
        output: str,
        system_id: str
    ) -> List[CausalClaim]:
        """
        Extract cause→effect pairs from subsystem output.
        
        Args:
            output: Text output from subsystem
            system_id: ID of subsystem
            
        Returns:
            List of causal claims
        """
        claims = []
        
        # Patterns for causal language
        causal_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)', 1.0),
            (r'because\s+(\w+(?:\s+\w+)*),\s+(\w+(?:\s+\w+)*)', 0.9),
            (r'(\w+(?:\s+\w+)*)\s+leads?\s+to\s+(\w+(?:\s+\w+)*)', 0.9),
            (r'if\s+(\w+(?:\s+\w+)*),?\s+then\s+(\w+(?:\s+\w+)*)', 0.8),
            (r'(\w+(?:\s+\w+)*)\s+results?\s+in\s+(\w+(?:\s+\w+)*)', 0.9),
            (r'due\s+to\s+(\w+(?:\s+\w+)*),\s+(\w+(?:\s+\w+)*)', 0.9),
        ]
        
        text_lower = output.lower()
        
        for pattern, confidence in causal_patterns:
            matches = re.findall(pattern, text_lower)
            for cause, effect in matches:
                # Clean up whitespace
                cause = ' '.join(cause.split())
                effect = ' '.join(effect.split())
                
                # Filter out very short or very long claims
                if 2 <= len(cause.split()) <= 10 and 2 <= len(effect.split()) <= 10:
                    claims.append(CausalClaim(
                        cause=cause,
                        effect=effect,
                        source_system=system_id,
                        confidence=confidence
                    ))
        
        return claims
    
    def _causal_agreement(
        self,
        claims_a: List[CausalClaim],
        claims_b: List[CausalClaim]
    ) -> float:
        """
        Compute agreement between two sets of causal claims.
        
        Args:
            claims_a: Claims from system A
            claims_b: Claims from system B
            
        Returns:
            Agreement score (0-1)
        """
        if not claims_a or not claims_b:
            return 0.5  # Neutral when one system has no claims
        
        # Convert to sets of (cause, effect) tuples
        set_a = {(c.cause, c.effect) for c in claims_a}
        set_b = {(c.cause, c.effect) for c in claims_b}
        
        # Exact matches
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union == 0:
            return 0.5
        
        # Jaccard similarity
        exact_agreement = intersection / union
        
        # Also check for semantic similarity (same cause or effect)
        cause_overlap = len({c for c, _ in set_a} & {c for c, _ in set_b})
        effect_overlap = len({e for _, e in set_a} & {e for _, e in set_b})
        
        total_causes = len({c for c, _ in set_a} | {c for c, _ in set_b})
        total_effects = len({e for _, e in set_a} | {e for _, e in set_b})
        
        cause_similarity = cause_overlap / total_causes if total_causes > 0 else 0
        effect_similarity = effect_overlap / total_effects if total_effects > 0 else 0
        
        # Weighted combination
        agreement = (
            exact_agreement * 0.6 +
            cause_similarity * 0.2 +
            effect_similarity * 0.2
        )
        
        return agreement
    
    def _compute_predictive_coherence(
        self,
        temporal_bindings: Optional[List[TemporalBinding]]
    ) -> float:
        """
        Compute predictive coherence.
        
        Measures how well the system predicts outcomes.
        
        Args:
            temporal_bindings: Recent temporal bindings
            
        Returns:
            Predictive coherence score (0-1)
        """
        if not temporal_bindings:
            return 0.7  # Default if no bindings
        
        # Count predictions that matched reality
        correct_predictions = 0
        total_predictions = 0
        
        for binding in temporal_bindings:
            if binding.outcome is not None:
                total_predictions += 1
                
                # Prediction is correct if coherence_delta is positive
                # (action improved coherence as expected)
                if binding.coherence_delta > 0:
                    correct_predictions += 1
        
        if total_predictions == 0:
            return 0.7
        
        predictive_coherence = correct_predictions / total_predictions
        
        # Track prediction accuracy
        self.prediction_history.append({
            'timestamp': time.time(),
            'accuracy': predictive_coherence,
            'sample_size': total_predictions
        })
        
        # Keep last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
        
        return predictive_coherence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coherence statistics."""
        stats = {
            'causal_agreement_history_size': len(self.causal_agreement_history),
            'avg_causal_agreement': (
                np.mean(self.causal_agreement_history)
                if self.causal_agreement_history else 0.0
            ),
            'prediction_history_size': len(self.prediction_history),
            'avg_predictive_accuracy': (
                np.mean([p['accuracy'] for p in self.prediction_history])
                if self.prediction_history else 0.0
            ),
        }
        
        if self.temporal_tracker:
            stats['temporal'] = self.temporal_tracker.get_statistics()
        
        return stats
