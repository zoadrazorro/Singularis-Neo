"""
Hebbian Integration System: "Neurons that fire together, wire together"

Tracks correlations between different AGI systems and strengthens connections
between systems that successfully collaborate.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class SystemActivation:
    """Records an instance of a system's activation.

    Attributes:
        system_name: The name of the system that was activated.
        timestamp: The time of the activation.
        success: A boolean indicating whether the activation contributed to a
                 successful outcome.
        contribution_strength: A float (0.0-1.0) indicating the perceived
                               contribution of this activation to the outcome.
        context: An optional dictionary for storing metadata about the activation.
    """
    system_name: str
    timestamp: float
    success: bool
    contribution_strength: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemCorrelation:
    """Tracks the learned correlation between two systems.

    Attributes:
        system_a: The name of the first system.
        system_b: The name of the second system.
        joint_activations: The total number of times these systems were active
                           within the same temporal window.
        joint_successes: The number of times their joint activation was successful.
        correlation_strength: The learned Hebbian weight between the two systems.
        last_updated: The timestamp of the last update to this correlation.
    """
    system_a: str
    system_b: str
    joint_activations: int = 0
    joint_successes: int = 0
    correlation_strength: float = 0.0
    last_updated: float = field(default_factory=time.time)


class HebbianIntegrator:
    """Implements Hebbian learning ("neurons that fire together, wire together")
    across different high-level AGI systems.

    This class tracks the co-activation of different modules (e.g., combat tactics,
    perception, emotion). When systems are active together within a defined time
    window and their activation leads to a successful outcome, the "synaptic"
    connection between them is strengthened. This allows the AGI to learn which
    systems are most effective when used in combination for certain tasks.
    """
    
    def __init__(
        self,
        temporal_window: float = 30.0,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
    ):
        """Initializes the HebbianIntegrator.

        Args:
            temporal_window: The time in seconds within which two system
                             activations are considered to be co-active.
            learning_rate: The rate at which correlation strengths and system
                           weights are updated.
            decay_rate: The rate at which all correlation strengths and weights
                        decay over time (simulating synaptic pruning).
        """
        self.temporal_window = temporal_window
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Track recent activations
        self.recent_activations: List[SystemActivation] = []
        
        # Correlation matrix between systems
        self.correlations: Dict[Tuple[str, str], SystemCorrelation] = {}
        
        # System weights (learned importance)
        self.system_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Success history
        self.success_history: List[Tuple[float, bool]] = []
        
        # Integration metrics
        self.total_activations = 0
        self.successful_integrations = 0
        
    def record_activation(
        self,
        system_name: str,
        success: bool,
        contribution_strength: float = 1.0,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records the activation of a system and updates the Hebbian network.

        This is the main entry point for the learning process.

        Args:
            system_name: The name of the system that was activated.
            success: Whether the outcome of the activation was successful.
            contribution_strength: The strength of the system's contribution.
            context: Optional metadata about the activation context.
        """
        activation = SystemActivation(
            system_name=system_name,
            timestamp=time.time(),
            success=success,
            contribution_strength=contribution_strength,
            context=context or {}
        )
        
        self.recent_activations.append(activation)
        self.total_activations += 1
        
        # Update correlations with recently active systems
        self._update_correlations(activation)
        
        # Update system weight based on success
        if success:
            self.system_weights[system_name] += self.learning_rate * contribution_strength
            self.successful_integrations += 1
        else:
            self.system_weights[system_name] -= self.learning_rate * 0.5 * contribution_strength
        
        # Normalize weights to prevent unbounded growth
        self._normalize_weights()
        
        # Clean old activations
        self._cleanup_old_activations()
        
    def _update_correlations(self, new_activation: SystemActivation) -> None:
        """Updates the correlation strengths based on a new system activation.

        Compares the new activation with all other activations in the recent
        temporal window and applies the Hebbian learning rule.

        Args:
            new_activation: The SystemActivation to process.
        """
        current_time = new_activation.timestamp
        
        # Find systems active within temporal window
        for other_activation in self.recent_activations:
            if other_activation.system_name == new_activation.system_name:
                continue
                
            time_diff = abs(current_time - other_activation.timestamp)
            if time_diff > self.temporal_window:
                continue
            
            # Create correlation key (alphabetically sorted for consistency)
            systems = tuple(sorted([new_activation.system_name, other_activation.system_name]))
            
            if systems not in self.correlations:
                self.correlations[systems] = SystemCorrelation(
                    system_a=systems[0],
                    system_b=systems[1]
                )
            
            corr = self.correlations[systems]
            corr.joint_activations += 1
            
            # Both successful? Strengthen correlation
            if new_activation.success and other_activation.success:
                corr.joint_successes += 1
                
                # Hebbian strengthening: proportional to recency
                recency_factor = 1.0 - (time_diff / self.temporal_window)
                strength_product = new_activation.contribution_strength * other_activation.contribution_strength
                
                delta = self.learning_rate * recency_factor * strength_product
                corr.correlation_strength += delta
            else:
                # Failure weakens correlation
                corr.correlation_strength -= self.learning_rate * 0.3
            
            # Clamp correlation strength
            corr.correlation_strength = max(0.0, min(2.0, corr.correlation_strength))
            corr.last_updated = current_time
            
    def _normalize_weights(self) -> None:
        """Normalizes the system importance weights to keep their average around 1.0."""
        if not self.system_weights:
            return
            
        total = sum(self.system_weights.values())
        if total > 0:
            avg = total / len(self.system_weights)
            # Keep average around 1.0
            if avg > 1.5:
                for system in self.system_weights:
                    self.system_weights[system] /= avg
                    
    def _cleanup_old_activations(self) -> None:
        """Removes activations from the history that are outside the temporal window."""
        current_time = time.time()
        self.recent_activations = [
            act for act in self.recent_activations
            if (current_time - act.timestamp) <= self.temporal_window
        ]
        
    def get_system_weight(self, system_name: str) -> float:
        """Gets the learned importance weight for a given system.

        Args:
            system_name: The name of the system.

        Returns:
            The learned weight, or a default of 1.0 if the system is unknown.
        """
        return self.system_weights.get(system_name, 1.0)
        
    def get_correlation(self, system_a: str, system_b: str) -> float:
        """Gets the learned correlation strength between two systems.

        Args:
            system_a: The name of the first system.
            system_b: The name of the second system.

        Returns:
            The correlation strength, or 0.0 if no correlation has been learned.
        """
        systems = tuple(sorted([system_a, system_b]))
        corr = self.correlations.get(systems)
        return corr.correlation_strength if corr else 0.0
        
    def get_synergistic_pairs(self, threshold: float = 1.0) -> List[Tuple[str, str, float]]:
        """Finds all pairs of systems with a correlation strength above a given threshold.

        Args:
            threshold: The minimum correlation strength to be considered synergistic.

        Returns:
            A list of tuples, each containing the two system names and their
            correlation strength, sorted from strongest to weakest.
        """
        pairs = []
        for (sys_a, sys_b), corr in self.correlations.items():
            if corr.correlation_strength >= threshold:
                pairs.append((sys_a, sys_b, corr.correlation_strength))
        return sorted(pairs, key=lambda x: x[2], reverse=True)
        
    def get_recommended_systems(
        self,
        active_system: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Recommends other systems that are highly correlated with a given active system.

        This can be used to decide which other modules to activate to support
        the currently active one.

        Args:
            active_system: The name of the system that is currently active.
            top_k: The maximum number of recommendations to return.

        Returns:
            A sorted list of tuples, each containing a recommended system name
            and its correlation strength with the active system.
        """
        recommendations = []
        
        for (sys_a, sys_b), corr in self.correlations.items():
            if sys_a == active_system:
                recommendations.append((sys_b, corr.correlation_strength))
            elif sys_b == active_system:
                recommendations.append((sys_a, corr.correlation_strength))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_k]
        
    def apply_hebbian_decay(self) -> None:
        """Applies a decay factor to all correlation strengths and system weights.

        This simulates synaptic pruning, where connections and neurons that are not
        regularly reinforced will gradually weaken and may be removed.
        """
        for corr in self.correlations.values():
            corr.correlation_strength *= (1.0 - self.decay_rate)
            
        for system in list(self.system_weights.keys()):
            self.system_weights[system] *= (1.0 - self.decay_rate)
            # Remove very weak weights
            if self.system_weights[system] < 0.1:
                del self.system_weights[system]
                
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieves a dictionary of key statistics about the Hebbian network.

        Returns:
            A dictionary containing metrics like total activations, success rate,
            and the number of tracked correlations.
        """
        success_rate = (
            self.successful_integrations / self.total_activations
            if self.total_activations > 0
            else 0.0
        )
        
        return {
            'total_activations': self.total_activations,
            'successful_integrations': self.successful_integrations,
            'success_rate': success_rate,
            'active_systems': len(self.system_weights),
            'correlations_tracked': len(self.correlations),
            'strongest_system': max(self.system_weights.items(), key=lambda x: x[1])[0] if self.system_weights else None,
            'strongest_weight': max(self.system_weights.values()) if self.system_weights else 0.0,
        }
        
    def print_status(self) -> None:
        """Prints a formatted, human-readable status report of the Hebbian network to the console."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("HEBBIAN INTEGRATION STATUS - Neurons that Fire Together, Wire Together")
        print("="*70)
        print(f"Total Activations: {stats['total_activations']}")
        print(f"Successful Integrations: {stats['successful_integrations']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Active Systems: {stats['active_systems']}")
        print(f"Correlations Tracked: {stats['correlations_tracked']}")
        
        if stats['strongest_system']:
            print(f"\nStrongest System: {stats['strongest_system']} (weight: {stats['strongest_weight']:.2f})")
        
        # Show top synergistic pairs
        pairs = self.get_synergistic_pairs(threshold=0.5)
        if pairs:
            print("\nTop Synergistic System Pairs:")
            for i, (sys_a, sys_b, strength) in enumerate(pairs[:5], 1):
                print(f"  {i}. {sys_a} ↔ {sys_b}: {strength:.2f}")
        
        # Show system weights
        if self.system_weights:
            print("\nSystem Importance Weights:")
            sorted_weights = sorted(self.system_weights.items(), key=lambda x: x[1], reverse=True)
            for system, weight in sorted_weights[:10]:
                print(f"  {system}: {weight:.2f}")
        
        print("="*70 + "\n")
