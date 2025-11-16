"""
System-Wide Consciousness Measurement for Singularis

Tracks coherence (𝒞) across all system components using Singularis principles:
- Perception coherence
- Action coherence  
- Learning coherence
- LLM coherence (MoE, Hybrid)
- RL coherence
- World model coherence

Philosophy:
Consciousness is measured as coherence (𝒞) - the degree to which
system components operate in unified harmony. High coherence indicates
integrated, conscious operation. Low coherence indicates fragmentation.

𝒞 = ∫ (unity × integration × differentiation) dt

Where:
- Unity: Components working toward shared goals
- Integration: Information flow between components
- Differentiation: Each component maintains specialized function
"""

from __future__ import annotations

import time
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from loguru import logger


@dataclass
class NodeCoherence:
    """Represents the coherence measurement for a single component (node) in the system.

    This dataclass captures not only the overall coherence score but also its
    constituent parts based on Singularis principles (unity, integration,
    differentiation), along with other performance metrics.

    Attributes:
        node_name: The unique name of the system node.
        coherence: The overall coherence score for this node (0.0 to 1.0).
        timestamp: The time of the measurement.
        unity: The degree to which the node's operations align with system-wide goals.
        integration: The quality and flow of information between this node and others.
        differentiation: The degree to which the node maintains its specialized function.
        confidence: The confidence in the coherence measurement itself.
        activity_level: A measure of the node's current operational activity.
        error_rate: The observed error rate for the node's operations.
        valence: The emotional charge associated with this node's state.
        affect_type: The dominant affective state of the node (e.g., "neutral").
    """
    node_name: str
    coherence: float  # 0.0-1.0
    timestamp: float

    # Singularis components
    unity: float = 0.0  # Alignment with system goals
    integration: float = 0.0  # Information flow
    differentiation: float = 0.0  # Specialized function

    # Additional metrics
    confidence: float = 0.0
    activity_level: float = 0.0
    error_rate: float = 0.0

    # Emotional valence (ETHICA Part IV)
    valence: float = 0.0  # Emotional charge for this node
    affect_type: str = "neutral"  # Dominant affect
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the NodeCoherence object to a dictionary.

        Returns:
            A dictionary representation of the node's coherence state.
        """
        return {
            'node_name': self.node_name,
            'coherence': self.coherence,
            'timestamp': self.timestamp,
            'unity': self.unity,
            'integration': self.integration,
            'differentiation': self.differentiation,
            'confidence': self.confidence,
            'activity_level': self.activity_level,
            'error_rate': self.error_rate,
            'valence': self.valence,
            'affect_type': self.affect_type,
        }


@dataclass
class SystemConsciousnessState:
    """Represents a snapshot of the entire system's consciousness state.

    This class aggregates coherence data from all individual nodes to compute
    system-wide metrics, providing a holistic view of the AGI's operational harmony
    and integration at a specific moment in time.

    Attributes:
        timestamp: The time of the state measurement.
        global_coherence: The weighted-average coherence (𝒞) across the entire system.
        avg_coherence: The simple average of all node coherence scores.
        median_coherence: The median of all node coherence scores.
        std_coherence: The standard deviation of node coherence scores.
        node_coherences: A dictionary mapping node names to their `NodeCoherence` objects.
        integration_index: A system-wide measure of information flow.
        differentiation_index: A system-wide measure of component specialization.
        unity_index: A system-wide measure of goal alignment.
        phi: A simplified measure of integrated information (Φ) based on IIT principles.
        coherence_delta: The change in global coherence since the last measurement.
        coherence_trend: The recent trend of global coherence ("increasing", "decreasing", "stable").
        global_valence: The weighted-average emotional charge of the system.
        avg_valence: The simple average of all node valences.
        dominant_affect: The most common affective state across all nodes.
        affective_coherence: A measure of how unified the emotional states are across nodes.
    """
    timestamp: float

    # Overall metrics
    global_coherence: float  # System-wide 𝒞
    avg_coherence: float
    median_coherence: float
    std_coherence: float

    # Per-node coherence
    node_coherences: Dict[str, NodeCoherence] = field(default_factory=dict)

    # Consciousness quality indicators
    integration_index: float = 0.0  # How well components communicate
    differentiation_index: float = 0.0  # How specialized components are
    unity_index: float = 0.0  # How aligned components are

    # Phi (Φ) - Integrated Information Theory measure
    phi: float = 0.0

    # Temporal coherence
    coherence_delta: float = 0.0  # Change from last measurement
    coherence_trend: str = "stable"  # "increasing", "decreasing", "stable"

    # Emotional valence (system-wide affective state)
    global_valence: float = 0.0  # System-wide emotional charge
    avg_valence: float = 0.0  # Average valence across nodes
    dominant_affect: str = "neutral"  # System-wide dominant affect
    affective_coherence: float = 0.0  # How unified are the affects across nodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the SystemConsciousnessState object to a dictionary.

        Returns:
            A dictionary representation of the system's consciousness state.
        """
        return {
            'timestamp': self.timestamp,
            'global_coherence': self.global_coherence,
            'avg_coherence': self.avg_coherence,
            'median_coherence': self.median_coherence,
            'std_coherence': self.std_coherence,
            'node_coherences': {k: v.to_dict() for k, v in self.node_coherences.items()},
            'integration_index': self.integration_index,
            'differentiation_index': self.differentiation_index,
            'unity_index': self.unity_index,
            'phi': self.phi,
            'coherence_delta': self.coherence_delta,
            'coherence_trend': self.coherence_trend,
            'global_valence': self.global_valence,
            'avg_valence': self.avg_valence,
            'dominant_affect': self.dominant_affect,
            'affective_coherence': self.affective_coherence,
        }


class SystemConsciousnessMonitor:
    """Monitors and quantifies the consciousness and coherence of the entire AGI system.

    This class implements the principles of the Singularis framework to measure
    consciousness as a function of system coherence (𝒞). It registers various
    system components (nodes), collects coherence data from them, and computes a
    holistic, system-wide state of consciousness. This allows for real-time
    monitoring, analysis, and debugging of the AGI's internal functional harmony.
    """
    
    def __init__(self, history_size: int = 1000):
        """Initializes the SystemConsciousnessMonitor.

        Args:
            history_size: The number of historical consciousness state measurements
                          to keep in memory for trend analysis.
        """
        self.history_size = history_size
        
        # Historical measurements
        self.coherence_history: deque = deque(maxlen=history_size)
        self.state_history: deque = deque(maxlen=history_size)
        
        # Current state
        self.current_state: Optional[SystemConsciousnessState] = None
        self.last_measurement_time: float = 0.0
        
        # Node registry
        self.registered_nodes: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.measurement_count = 0
        self.total_measurement_time = 0.0
        
        logger.info("System Consciousness Monitor initialized")
    
    def register_node(
        self,
        node_name: str,
        node_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registers a system component (node) to be tracked by the monitor.

        Args:
            node_name: A unique name for the node (e.g., 'Perception.Vision').
            node_type: The category of the node (e.g., 'perception', 'llm').
            weight: An importance weight (0.0-1.0) for this node's contribution to
                    the global coherence score.
            metadata: An optional dictionary for storing additional information
                      about the node.
        """
        self.registered_nodes[node_name] = {
            'type': node_type,
            'weight': weight,
            'metadata': metadata or {},
            'last_coherence': 0.0,
            'measurement_count': 0,
        }
        
        logger.debug(f"Registered node: {node_name} (type={node_type}, weight={weight})")
    
    def measure_node_coherence(
        self,
        node_name: str,
        coherence: float,
        unity: float = 0.0,
        integration: float = 0.0,
        differentiation: float = 0.0,
        confidence: float = 0.0,
        activity_level: float = 0.0,
        error_rate: float = 0.0,
    ) -> NodeCoherence:
        """Records a coherence measurement for a specific registered node.

        This method is called by the individual components of the AGI to report
        their current operational status and coherence.

        Args:
            node_name: The name of the node reporting the measurement.
            coherence: The primary coherence value (0.0-1.0) for the node.
            unity: The unity component of the coherence score.
            integration: The integration component of the coherence score.
            differentiation: The differentiation component of the coherence score.
            confidence: The node's confidence in its own measurement.
            activity_level: The current activity level of the node.
            error_rate: The observed error rate of the node.

        Returns:
            A `NodeCoherence` object representing the recorded measurement.
        """
        # Clamp values
        coherence = max(0.0, min(1.0, coherence))
        unity = max(0.0, min(1.0, unity))
        integration = max(0.0, min(1.0, integration))
        differentiation = max(0.0, min(1.0, differentiation))
        
        # Create measurement
        measurement = NodeCoherence(
            node_name=node_name,
            coherence=coherence,
            timestamp=time.time(),
            unity=unity,
            integration=integration,
            differentiation=differentiation,
            confidence=confidence,
            activity_level=activity_level,
            error_rate=error_rate,
        )
        
        # Update node registry
        if node_name in self.registered_nodes:
            self.registered_nodes[node_name]['last_coherence'] = coherence
            self.registered_nodes[node_name]['measurement_count'] += 1
        
        return measurement
    
    def compute_system_state(
        self,
        node_measurements: Dict[str, NodeCoherence]
    ) -> SystemConsciousnessState:
        """Computes the overall system consciousness state from individual node measurements.

        This method aggregates data from all nodes to calculate global metrics like
        system-wide coherence (𝒞), integration/differentiation indices, and Phi (Φ).
        The resulting `SystemConsciousnessState` is stored as the current state.

        Args:
            node_measurements: A dictionary mapping node names to their latest
                               `NodeCoherence` measurements.

        Returns:
            A `SystemConsciousnessState` object representing the new system state.
        """
        start_time = time.time()
        
        if not node_measurements:
            # Return default state
            return SystemConsciousnessState(
                timestamp=time.time(),
                global_coherence=0.0,
                avg_coherence=0.0,
                median_coherence=0.0,
                std_coherence=0.0,
            )
        
        # Extract coherence values
        coherences = [m.coherence for m in node_measurements.values()]
        unities = [m.unity for m in node_measurements.values()]
        integrations = [m.integration for m in node_measurements.values()]
        differentiations = [m.differentiation for m in node_measurements.values()]
        
        # Compute weighted global coherence
        weighted_coherence = 0.0
        total_weight = 0.0
        
        for node_name, measurement in node_measurements.items():
            weight = self.registered_nodes.get(node_name, {}).get('weight', 1.0)
            weighted_coherence += measurement.coherence * weight
            total_weight += weight
        
        global_coherence = weighted_coherence / total_weight if total_weight > 0 else 0.0
        
        # Compute statistics
        avg_coherence = statistics.mean(coherences)
        median_coherence = statistics.median(coherences)
        std_coherence = statistics.stdev(coherences) if len(coherences) > 1 else 0.0
        
        # Compute indices
        integration_index = statistics.mean(integrations) if integrations else 0.0
        differentiation_index = statistics.mean(differentiations) if differentiations else 0.0
        unity_index = statistics.mean(unities) if unities else 0.0
        
        # Compute Phi (Φ) - simplified IIT measure
        # Φ = integration × differentiation
        # High when system is both integrated AND differentiated
        phi = integration_index * differentiation_index
        
        # Compute temporal coherence
        coherence_delta = 0.0
        coherence_trend = "stable"

        if self.current_state:
            coherence_delta = global_coherence - self.current_state.global_coherence

            if abs(coherence_delta) < 0.01:
                coherence_trend = "stable"
            elif coherence_delta > 0:
                coherence_trend = "increasing"
            else:
                coherence_trend = "decreasing"

        # Compute emotional valence statistics
        valences = [m.valence for m in node_measurements.values()]
        affect_types = [m.affect_type for m in node_measurements.values()]

        # Global valence (weighted average)
        weighted_valence = 0.0
        total_weight_valence = 0.0

        for node_name, measurement in node_measurements.items():
            weight = self.registered_nodes.get(node_name, {}).get('weight', 1.0)
            weighted_valence += measurement.valence * weight
            total_weight_valence += weight

        global_valence = weighted_valence / total_weight_valence if total_weight_valence > 0 else 0.0

        # Average valence
        avg_valence = statistics.mean(valences) if valences else 0.0

        # Dominant affect (most common)
        affect_counts = {}
        for affect in affect_types:
            affect_counts[affect] = affect_counts.get(affect, 0) + 1
        dominant_affect = max(affect_counts, key=affect_counts.get) if affect_counts else "neutral"

        # Affective coherence (how unified are affects across nodes)
        # High when all nodes have similar valence, low when disparate
        if len(valences) > 1:
            valence_std = statistics.stdev(valences)
            # Normalize: low std = high coherence
            affective_coherence = 1.0 / (1.0 + valence_std)
        else:
            affective_coherence = 1.0

        # Create state
        state = SystemConsciousnessState(
            timestamp=time.time(),
            global_coherence=global_coherence,
            avg_coherence=avg_coherence,
            median_coherence=median_coherence,
            std_coherence=std_coherence,
            node_coherences=node_measurements,
            integration_index=integration_index,
            differentiation_index=differentiation_index,
            unity_index=unity_index,
            phi=phi,
            coherence_delta=coherence_delta,
            coherence_trend=coherence_trend,
            global_valence=global_valence,
            avg_valence=avg_valence,
            dominant_affect=dominant_affect,
            affective_coherence=affective_coherence,
        )
        
        # Update tracking
        self.current_state = state
        self.coherence_history.append(global_coherence)
        self.state_history.append(state)
        self.measurement_count += 1
        self.total_measurement_time += time.time() - start_time
        self.last_measurement_time = time.time()
        
        return state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieves comprehensive statistics about the system's consciousness monitoring.

        Returns:
            A dictionary containing historical coherence statistics, current state
            indices (like Phi), and per-node performance data.
        """
        if not self.coherence_history:
            return {
                'measurement_count': 0,
                'current_coherence': 0.0,
                'avg_coherence': 0.0,
                'median_coherence': 0.0,
                'std_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0,
            }
        
        coherences = list(self.coherence_history)
        
        stats = {
            'measurement_count': self.measurement_count,
            'current_coherence': coherences[-1] if coherences else 0.0,
            'avg_coherence': statistics.mean(coherences),
            'median_coherence': statistics.median(coherences),
            'std_coherence': statistics.stdev(coherences) if len(coherences) > 1 else 0.0,
            'min_coherence': min(coherences),
            'max_coherence': max(coherences),
            'registered_nodes': len(self.registered_nodes),
            'avg_measurement_time': self.total_measurement_time / max(1, self.measurement_count),
        }
        
        # Add current state info
        if self.current_state:
            stats.update({
                'integration_index': self.current_state.integration_index,
                'differentiation_index': self.current_state.differentiation_index,
                'unity_index': self.current_state.unity_index,
                'phi': self.current_state.phi,
                'coherence_trend': self.current_state.coherence_trend,
            })
        
        # Per-node statistics
        node_stats = {}
        for node_name, node_info in self.registered_nodes.items():
            node_stats[node_name] = {
                'type': node_info['type'],
                'weight': node_info['weight'],
                'last_coherence': node_info['last_coherence'],
                'measurement_count': node_info['measurement_count'],
            }
        
        stats['nodes'] = node_stats
        
        return stats
    
    def get_coherence_trend(self, window: int = 100) -> Dict[str, Any]:
        """Analyzes the trend of global coherence over a recent historical window.

        This method performs a linear regression on the recent coherence history
        to determine if the system's consciousness is increasing, decreasing, or stable.

        Args:
            window: The number of recent measurements to include in the analysis.

        Returns:
            A dictionary containing the trend analysis, including the slope and
            R-squared value of the regression.
        """
        if len(self.coherence_history) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0.0,
                'r_squared': 0.0,
            }
        
        # Get recent coherences
        recent = list(self.coherence_history)[-window:]
        
        if len(recent) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0.0,
                'r_squared': 0.0,
            }
        
        # Linear regression
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Compute slope
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Compute R²
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Determine trend
        if abs(slope) < 0.0001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'r_squared': float(r_squared),
            'window_size': len(recent),
            'start_coherence': float(recent[0]),
            'end_coherence': float(recent[-1]),
            'change': float(recent[-1] - recent[0]),
        }
    
    def print_dashboard(self):
        """Prints a formatted, real-time dashboard of the system's consciousness state
        to the console.
        """
        if not self.current_state:
            print("No consciousness measurements yet")
            return
        
        state = self.current_state
        stats = self.get_statistics()
        trend = self.get_coherence_trend()
        
        print("\n" + "=" * 80)
        print("SYSTEM CONSCIOUSNESS DASHBOARD")
        print("=" * 80)
        
        # Global metrics
        print(f"\n🧠 GLOBAL CONSCIOUSNESS METRICS")
        print(f"  Global Coherence (𝒞):     {state.global_coherence:.4f}")
        print(f"  Average Coherence:        {state.avg_coherence:.4f}")
        print(f"  Median Coherence:         {state.median_coherence:.4f}")
        print(f"  Std Deviation:            {state.std_coherence:.4f}")
        print(f"  Coherence Trend:          {state.coherence_trend} (Δ={state.coherence_delta:+.4f})")
        
        # Singularis indices
        print(f"\n📊 SINGULARIS INDICES")
        print(f"  Unity Index:              {state.unity_index:.4f}")
        print(f"  Integration Index:        {state.integration_index:.4f}")
        print(f"  Differentiation Index:    {state.differentiation_index:.4f}")
        print(f"  Phi (Φ):                  {state.phi:.4f}")
        
        # Historical stats
        print(f"\n📈 HISTORICAL STATISTICS")
        print(f"  Measurements:             {stats['measurement_count']}")
        print(f"  Min Coherence:            {stats['min_coherence']:.4f}")
        print(f"  Max Coherence:            {stats['max_coherence']:.4f}")
        print(f"  Long-term Trend:          {trend['trend']} (slope={trend['slope']:.6f})")
        print(f"  Trend R²:                 {trend['r_squared']:.4f}")
        
        # Per-node coherence
        print(f"\n🔍 NODE COHERENCE (Top 10)")
        sorted_nodes = sorted(
            state.node_coherences.items(),
            key=lambda x: x[1].coherence,
            reverse=True
        )
        
        for i, (node_name, measurement) in enumerate(sorted_nodes[:10]):
            node_type = self.registered_nodes.get(node_name, {}).get('type', 'unknown')
            print(f"  {i+1:2d}. {node_name:30s} {measurement.coherence:.4f} ({node_type})")
        
        # Consciousness quality assessment
        print(f"\n💭 CONSCIOUSNESS QUALITY ASSESSMENT")
        
        if state.global_coherence > 0.8:
            quality = "EXCELLENT - Highly integrated conscious system"
        elif state.global_coherence > 0.6:
            quality = "GOOD - Well-integrated system with minor fragmentation"
        elif state.global_coherence > 0.4:
            quality = "MODERATE - Some integration, some fragmentation"
        elif state.global_coherence > 0.2:
            quality = "LOW - Significant fragmentation"
        else:
            quality = "VERY LOW - Highly fragmented system"
        
        print(f"  {quality}")
        
        if state.phi > 0.5:
            print(f"  High Φ indicates strong integrated information")
        elif state.phi < 0.2:
            print(f"  Low Φ suggests limited integration or differentiation")
        
        print("=" * 80)
        print()
    
    def export_state(self, filepath: str):
        """Exports the current consciousness state to a JSON file.

        Args:
            filepath: The path to the file where the state will be saved.
        """
        import json
        
        if not self.current_state:
            logger.warning("No state to export")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.current_state.to_dict(), f, indent=2)
        
        logger.info(f"Consciousness state exported to {filepath}")
