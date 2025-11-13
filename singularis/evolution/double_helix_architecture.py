"""
Double-Helix Architecture with Self-Improvement Gating

Interconnects ALL systems in a double-helix pattern where:
- Strand 1: Analytical/Logical systems (left helix)
- Strand 2: Intuitive/Evolutionary systems (right helix)
- Base pairs: Cross-connections between strands
- Self-improvement gating: Nodes with higher integration get weighted more

Systems integrated:
1. Sensorimotor (Claude 4.5)
2. Emotion (HuiHui)
3. Spiritual Awareness
4. Symbolic Logic
5. Action Planning
6. World Model
7. Consciousness Bridge
8. Hebbian Integration
9. Self-Reflection (GPT-4 Realtime)
10. Reward Tuning (Claude Sonnet 4.5)
11. Realtime Coordinator (GPT-4 Realtime)
12. Darwinian Modal Logic (Gemini Flash 2.0)
13. Analytic Evolution (Claude Haiku)
14. Voice System (Gemini 2.5 Pro TTS)
15. Streaming Video Interpreter (Gemini 2.5 Flash Native Audio)
16. Hyperbolic Reasoning (Qwen3-235B) - Meta-cognitive reasoning
17. Hyperbolic Vision (NVIDIA Nemotron) - Visual awareness

Architecture:
- Each system is a node
- Integration score = number of connections to other systems
- Reward gating = nodes with higher integration get higher weight in decisions
- Double helix = analytical and intuitive strands intertwine
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import numpy as np
from loguru import logger


class SystemStrand(Enum):
    """Which helix strand a system belongs to."""
    ANALYTICAL = "analytical"  # Left helix - logical, symbolic
    INTUITIVE = "intuitive"  # Right helix - evolutionary, emotional


@dataclass
class SystemNode:
    """
    A node in the double-helix architecture.
    
    Represents one AGI subsystem.
    """
    node_id: str
    name: str
    strand: SystemStrand
    
    # Connections
    connections: Set[str] = field(default_factory=set)  # Connected node IDs
    base_pairs: Set[str] = field(default_factory=set)  # Cross-strand connections
    
    # Integration metrics
    integration_score: float = 0.0  # How well integrated with other systems
    contribution_weight: float = 1.0  # Weight in decision-making
    
    # Self-improvement tracking
    improvement_rate: float = 0.0  # Rate of improvement over time
    total_activations: int = 0
    successful_activations: int = 0
    
    # Gating
    is_gated: bool = False  # Whether node is currently gated
    gate_threshold: float = 0.5  # Minimum integration for ungating
    
    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.total_activations == 0:
            return 0.0
        return self.successful_activations / self.total_activations
    
    def compute_integration_score(self, total_nodes: int) -> float:
        """Compute integration score based on connections."""
        if total_nodes <= 1:
            return 0.0
        
        # Integration = (connections + base_pairs) / (total_nodes - 1)
        total_connections = len(self.connections) + len(self.base_pairs)
        max_connections = total_nodes - 1
        
        self.integration_score = total_connections / max_connections
        return self.integration_score
    
    def update_contribution_weight(self):
        """Update contribution weight based on integration and performance."""
        # Weight = integration_score * success_rate * (1 + improvement_rate)
        self.contribution_weight = (
            self.integration_score *
            self.success_rate *
            (1.0 + self.improvement_rate)
        )
        
        # Gate if integration too low
        self.is_gated = self.integration_score < self.gate_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'name': self.name,
            'strand': self.strand.value,
            'connections': len(self.connections),
            'base_pairs': len(self.base_pairs),
            'integration_score': float(self.integration_score),
            'contribution_weight': float(self.contribution_weight),
            'success_rate': float(self.success_rate),
            'is_gated': self.is_gated
        }


class DoubleHelixArchitecture:
    """
    Double-helix architecture interconnecting all AGI systems.
    
    Implements self-improvement gating where highly integrated nodes
    have more influence on decisions.
    """
    
    def __init__(self):
        """Initialize double-helix architecture."""
        # Nodes
        self.nodes: Dict[str, SystemNode] = {}
        
        # Helix strands
        self.analytical_strand: List[str] = []  # Left helix
        self.intuitive_strand: List[str] = []  # Right helix
        
        # Statistics
        self.total_activations = 0
        self.total_integrations = 0
        
        logger.info("[DOUBLE-HELIX] Architecture initialized")
    
    def initialize_systems(self):
        """Initialize all system nodes in double-helix pattern."""
        
        # ANALYTICAL STRAND (Left Helix)
        analytical_systems = [
            ("symbolic_logic", "Symbolic Logic World Model"),
            ("action_planning", "Action Planning"),
            ("world_model", "World Model"),
            ("consciousness", "Consciousness Bridge"),
            ("analytic_evolution", "Analytic Evolution (Claude Haiku)"),
            ("reward_tuning", "Reward-Guided Tuning (Claude Sonnet 4.5)"),
            ("darwinian_logic", "Darwinian Modal Logic (Gemini Flash 2.0)"),
            ("hyperbolic_reasoning", "Hyperbolic Reasoning (Qwen3-235B)")
        ]
        
        for node_id, name in analytical_systems:
            node = SystemNode(
                node_id=node_id,
                name=name,
                strand=SystemStrand.ANALYTICAL
            )
            self.nodes[node_id] = node
            self.analytical_strand.append(node_id)
        
        # INTUITIVE STRAND (Right Helix)
        intuitive_systems = [
            ("sensorimotor", "Sensorimotor (Claude 4.5)"),
            ("emotion", "Emotion System (HuiHui)"),
            ("spiritual", "Spiritual Awareness"),
            ("hebbian", "Hebbian Integration"),
            ("self_reflection", "Self-Reflection (GPT-4 Realtime)"),
            ("realtime_coordinator", "Realtime Coordinator (GPT-4 Realtime)"),
            ("voice_system", "Voice System (Gemini 2.5 Pro TTS)"),
            ("video_interpreter", "Streaming Video Interpreter (Gemini 2.5 Flash Native Audio)"),
            ("hyperbolic_vision", "Hyperbolic Vision (NVIDIA Nemotron)")
        ]
        
        for node_id, name in intuitive_systems:
            node = SystemNode(
                node_id=node_id,
                name=name,
                strand=SystemStrand.INTUITIVE
            )
            self.nodes[node_id] = node
            self.intuitive_strand.append(node_id)
        
        # Create double-helix connections
        self._create_helix_connections()
        
        # Compute initial integration scores
        self._update_all_integration_scores()
        
        logger.info(f"[DOUBLE-HELIX] Initialized {len(self.nodes)} systems")
        logger.info(f"[DOUBLE-HELIX] Analytical strand: {len(self.analytical_strand)} nodes")
        logger.info(f"[DOUBLE-HELIX] Intuitive strand: {len(self.intuitive_strand)} nodes")
    
    def _create_helix_connections(self):
        """Create double-helix connection pattern."""
        
        # 1. Connect within each strand (backbone)
        for i in range(len(self.analytical_strand) - 1):
            node1 = self.analytical_strand[i]
            node2 = self.analytical_strand[i + 1]
            self._connect_nodes(node1, node2)
        
        for i in range(len(self.intuitive_strand) - 1):
            node1 = self.intuitive_strand[i]
            node2 = self.intuitive_strand[i + 1]
            self._connect_nodes(node1, node2)
        
        # 2. Create base pairs (cross-strand connections)
        # Each analytical node connects to corresponding intuitive node
        max_pairs = min(len(self.analytical_strand), len(self.intuitive_strand))
        
        for i in range(max_pairs):
            analytical_node = self.analytical_strand[i]
            intuitive_node = self.intuitive_strand[i]
            self._create_base_pair(analytical_node, intuitive_node)
        
        # 3. Create additional cross-connections for full integration
        # Every system connects to at least 3 others from opposite strand
        for analytical_id in self.analytical_strand:
            # Connect to 3 intuitive systems
            for intuitive_id in self.intuitive_strand[:3]:
                if intuitive_id not in self.nodes[analytical_id].base_pairs:
                    self._create_base_pair(analytical_id, intuitive_id)
        
        for intuitive_id in self.intuitive_strand:
            # Connect to 3 analytical systems
            for analytical_id in self.analytical_strand[:3]:
                if analytical_id not in self.nodes[intuitive_id].base_pairs:
                    self._create_base_pair(intuitive_id, analytical_id)
        
        logger.info("[DOUBLE-HELIX] Created helix connections")
    
    def _connect_nodes(self, node1_id: str, node2_id: str):
        """Connect two nodes (same-strand connection)."""
        self.nodes[node1_id].connections.add(node2_id)
        self.nodes[node2_id].connections.add(node1_id)
    
    def _create_base_pair(self, node1_id: str, node2_id: str):
        """Create base pair (cross-strand connection)."""
        self.nodes[node1_id].base_pairs.add(node2_id)
        self.nodes[node2_id].base_pairs.add(node1_id)
    
    def _update_all_integration_scores(self):
        """Update integration scores for all nodes."""
        total_nodes = len(self.nodes)
        
        for node in self.nodes.values():
            node.compute_integration_score(total_nodes)
            node.update_contribution_weight()
    
    def record_activation(
        self,
        node_id: str,
        success: bool,
        contribution: float = 1.0
    ):
        """
        Record system activation.
        
        Args:
            node_id: System that was activated
            success: Whether activation was successful
            contribution: Contribution strength (0.0 to 1.0)
        """
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.total_activations += 1
        
        if success:
            node.successful_activations += 1
        
        # Update improvement rate
        old_success_rate = node.success_rate
        new_success_rate = node.successful_activations / node.total_activations
        node.improvement_rate = new_success_rate - old_success_rate
        
        # Update contribution weight
        node.update_contribution_weight()
        
        self.total_activations += 1
    
    def get_weighted_contributions(self) -> Dict[str, float]:
        """
        Get contribution weights for all systems.
        
        Returns dict of {node_id: weight} where higher weight = more influence.
        """
        weights = {}
        
        for node_id, node in self.nodes.items():
            if node.is_gated:
                weights[node_id] = 0.0  # Gated nodes have no weight
            else:
                weights[node_id] = node.contribution_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_top_contributors(self, limit: int = 5) -> List[SystemNode]:
        """Get top contributing systems."""
        # Filter out gated nodes
        active_nodes = [n for n in self.nodes.values() if not n.is_gated]
        
        # Sort by contribution weight
        sorted_nodes = sorted(
            active_nodes,
            key=lambda n: n.contribution_weight,
            reverse=True
        )
        
        return sorted_nodes[:limit]
    
    def integrate_decision(
        self,
        subsystem_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate outputs from multiple subsystems using weighted contributions.
        
        Args:
            subsystem_outputs: Dict of {node_id: output}
        
        Returns:
            Integrated decision
        """
        self.total_integrations += 1
        
        # Get weights
        weights = self.get_weighted_contributions()
        
        # Weighted integration
        integrated = {
            'subsystem_outputs': subsystem_outputs,
            'weights': weights,
            'top_contributors': []
        }
        
        # Identify top contributors
        for node_id, output in subsystem_outputs.items():
            if node_id in weights and weights[node_id] > 0:
                integrated['top_contributors'].append({
                    'system': self.nodes[node_id].name,
                    'weight': weights[node_id],
                    'output': str(output)[:100]
                })
        
        # Sort by weight
        integrated['top_contributors'].sort(
            key=lambda x: x['weight'],
            reverse=True
        )
        
        return integrated
    
    def visualize_helix(self) -> str:
        """Generate ASCII visualization of double helix."""
        lines = []
        lines.append("DOUBLE-HELIX ARCHITECTURE")
        lines.append("=" * 70)
        lines.append("")
        
        # Show both strands side by side
        max_len = max(len(self.analytical_strand), len(self.intuitive_strand))
        
        for i in range(max_len):
            # Analytical (left)
            if i < len(self.analytical_strand):
                analytical_id = self.analytical_strand[i]
                analytical_node = self.nodes[analytical_id]
                analytical_str = f"{analytical_node.name[:25]:25} [{analytical_node.contribution_weight:.2f}]"
            else:
                analytical_str = " " * 35
            
            # Intuitive (right)
            if i < len(self.intuitive_strand):
                intuitive_id = self.intuitive_strand[i]
                intuitive_node = self.nodes[intuitive_id]
                intuitive_str = f"[{intuitive_node.contribution_weight:.2f}] {intuitive_node.name[:25]:25}"
            else:
                intuitive_str = ""
            
            # Connection symbol
            if i < min(len(self.analytical_strand), len(self.intuitive_strand)):
                connector = " === "
            else:
                connector = "     "
            
            lines.append(f"{analytical_str}{connector}{intuitive_str}")
        
        lines.append("")
        lines.append("Legend: [weight] = contribution weight (higher = more influence)")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get architecture statistics."""
        total_connections = sum(len(n.connections) for n in self.nodes.values())
        total_base_pairs = sum(len(n.base_pairs) for n in self.nodes.values()) // 2
        
        avg_integration = sum(n.integration_score for n in self.nodes.values()) / len(self.nodes)
        avg_weight = sum(n.contribution_weight for n in self.nodes.values()) / len(self.nodes)
        
        gated_count = sum(1 for n in self.nodes.values() if n.is_gated)
        
        return {
            'total_nodes': len(self.nodes),
            'analytical_nodes': len(self.analytical_strand),
            'intuitive_nodes': len(self.intuitive_strand),
            'total_connections': total_connections,
            'total_base_pairs': total_base_pairs,
            'average_integration': float(avg_integration),
            'average_weight': float(avg_weight),
            'gated_nodes': gated_count,
            'total_activations': self.total_activations,
            'total_integrations': self.total_integrations
        }
    
    # Helper methods for new systems
    
    def record_voice_activation(self, thought_spoken: bool, priority: str):
        """Record voice system activation."""
        self.record_activation(
            node_id="voice_system",
            success=thought_spoken,
            contribution=1.0 if priority in ["HIGH", "CRITICAL"] else 0.5
        )
    
    def record_video_interpretation(self, interpretation_success: bool, mode: str):
        """Record video interpreter activation."""
        self.record_activation(
            node_id="video_interpreter",
            success=interpretation_success,
            contribution=1.0 if mode == "COMPREHENSIVE" else 0.7
        )
