"""
Modular Network Architecture - Universal Foundation

Every component in Singularis (Meta-MoE agents, LLMs, swarm nodes, neurons, logic gates)
is built on this modular network foundation with:

1. Scale-free networks: Power-law degree distribution with hub nodes
2. Small-world networks: High clustering with short path lengths  
3. Modular networks: Dense intra-module, sparse inter-module connections

This provides brain-like topology for all system components.
"""

from __future__ import annotations

import random
import math
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
from loguru import logger


class NetworkTopology(Enum):
    """Network topology types."""
    SCALE_FREE = "scale_free"        # Power-law degree distribution
    SMALL_WORLD = "small_world"      # High clustering + short paths
    MODULAR = "modular"              # Module-based organization
    HYBRID = "hybrid"                # Combination of all three


class ModuleType(Enum):
    """Module specialization types."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    ACTION = "action"
    EMOTION = "emotion"
    COORDINATION = "coordination"
    LEARNING = "learning"
    SYNTHESIS = "synthesis"


@dataclass
class NetworkNode:
    """Universal network node (agent, neuron, LLM, logic gate, etc.)."""
    node_id: int
    node_type: str                   # "agent", "neuron", "llm", "gate", etc.
    module_id: int                   # Which module this belongs to
    module_type: ModuleType
    
    # Network properties
    connections: Dict[int, float] = field(default_factory=dict)  # node_id -> weight
    degree: int = 0                  # Number of connections
    clustering_coeff: float = 0.0    # Local clustering coefficient
    betweenness: float = 0.0         # Betweenness centrality
    
    # Node state
    activation: float = 0.0          # Current activation [0, 1]
    is_hub: bool = False             # Whether this is a hub node
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_connection(self, target_id: int, weight: float = 0.5):
        """Add connection to another node."""
        self.connections[target_id] = weight
        self.degree = len(self.connections)
    
    def remove_connection(self, target_id: int):
        """Remove connection to another node."""
        if target_id in self.connections:
            del self.connections[target_id]
            self.degree = len(self.connections)
    
    def update_weight(self, target_id: int, delta: float):
        """Update connection weight."""
        if target_id in self.connections:
            self.connections[target_id] = np.clip(
                self.connections[target_id] + delta,
                0.0, 1.0
            )


@dataclass
class NetworkModule:
    """Module containing related nodes."""
    module_id: int
    module_type: ModuleType
    nodes: Set[int] = field(default_factory=set)
    
    # Module properties
    internal_density: float = 0.0    # Density of intra-module connections
    external_connections: int = 0    # Number of inter-module connections
    
    def add_node(self, node_id: int):
        """Add node to module."""
        self.nodes.add(node_id)
    
    def remove_node(self, node_id: int):
        """Remove node from module."""
        self.nodes.discard(node_id)


class ModularNetwork:
    """
    Universal modular network architecture.
    
    Provides brain-like topology for any component type:
    - Meta-MoE agents
    - LLM instances
    - Swarm intelligence nodes
    - Bio-simulator neurons
    - Logic gates
    - Any other component
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_modules: int,
        topology: NetworkTopology = NetworkTopology.HYBRID,
        node_type: str = "generic",
        
        # Scale-free parameters
        preferential_attachment_m: int = 2,
        
        # Small-world parameters
        small_world_k: int = 4,
        small_world_p: float = 0.1,
        
        # Modular parameters
        intra_module_density: float = 0.3,
        inter_module_density: float = 0.05,
    ):
        """
        Initialize modular network.
        
        Args:
            num_nodes: Total number of nodes
            num_modules: Number of modules
            topology: Network topology type
            node_type: Type of nodes ("agent", "neuron", "llm", etc.)
            preferential_attachment_m: Edges per node in scale-free
            small_world_k: Neighbors in small-world
            small_world_p: Rewiring probability in small-world
            intra_module_density: Connection density within modules
            inter_module_density: Connection density between modules
        """
        self.num_nodes = num_nodes
        self.num_modules = num_modules
        self.topology = topology
        self.node_type = node_type
        
        # Parameters
        self.pa_m = preferential_attachment_m
        self.sw_k = small_world_k
        self.sw_p = small_world_p
        self.intra_density = intra_module_density
        self.inter_density = inter_module_density
        
        # Network components
        self.nodes: Dict[int, NetworkNode] = {}
        self.modules: Dict[int, NetworkModule] = {}
        
        # Hub nodes (high degree)
        self.hub_nodes: Set[int] = set()
        
        # Statistics
        self.stats = {
            'avg_degree': 0.0,
            'avg_clustering': 0.0,
            'avg_path_length': 0.0,
            'modularity': 0.0,
            'num_hubs': 0,
            'scale_free_exponent': 0.0,
        }
        
        # Build network
        self._initialize_modules()
        self._initialize_nodes()
        self._build_topology()
        self._identify_hubs()
        self._compute_statistics()
        
        logger.info(
            f"[MODULAR-NET] Initialized {num_nodes} {node_type} nodes in "
            f"{num_modules} modules with {topology.value} topology"
        )
    
    def _initialize_modules(self):
        """Initialize network modules."""
        module_types = list(ModuleType)
        
        for i in range(self.num_modules):
            # Assign module type (cycle through types)
            mod_type = module_types[i % len(module_types)]
            
            self.modules[i] = NetworkModule(
                module_id=i,
                module_type=mod_type
            )
    
    def _initialize_nodes(self):
        """Initialize network nodes and assign to modules."""
        nodes_per_module = self.num_nodes // self.num_modules
        
        for i in range(self.num_nodes):
            # Assign to module (roughly balanced)
            module_id = i // nodes_per_module
            if module_id >= self.num_modules:
                module_id = self.num_modules - 1
            
            module = self.modules[module_id]
            
            self.nodes[i] = NetworkNode(
                node_id=i,
                node_type=self.node_type,
                module_id=module_id,
                module_type=module.module_type
            )
            
            module.add_node(i)
    
    def _build_topology(self):
        """Build network topology based on type."""
        if self.topology == NetworkTopology.SCALE_FREE:
            self._build_scale_free()
        elif self.topology == NetworkTopology.SMALL_WORLD:
            self._build_small_world()
        elif self.topology == NetworkTopology.MODULAR:
            self._build_modular()
        else:  # HYBRID
            self._build_hybrid()
    
    def _build_scale_free(self):
        """
        Build scale-free network using Barabási-Albert preferential attachment.
        
        Power-law degree distribution: P(k) ~ k^(-γ)
        """
        m0 = max(3, self.pa_m)  # Initial complete graph size
        m = self.pa_m           # Edges to attach per new node
        
        # Start with complete graph
        for i in range(m0):
            for j in range(i + 1, m0):
                weight = random.uniform(0.3, 0.7)
                self.nodes[i].add_connection(j, weight)
                self.nodes[j].add_connection(i, weight)
        
        # Preferential attachment
        for i in range(m0, self.num_nodes):
            # Calculate connection probabilities (proportional to degree)
            degrees = {j: self.nodes[j].degree for j in range(i)}
            total_degree = sum(degrees.values())
            
            if total_degree == 0:
                # Fallback: random connections
                targets = random.sample(range(i), min(m, i))
            else:
                # Preferential attachment
                probs = {j: deg / total_degree for j, deg in degrees.items()}
                targets = []
                for _ in range(min(m, i)):
                    remaining = [j for j in probs.keys() if j not in targets]
                    if not remaining:
                        break
                    target = random.choices(
                        remaining,
                        weights=[probs[j] for j in remaining]
                    )[0]
                    targets.append(target)
            
            # Create connections
            for j in targets:
                weight = random.uniform(0.3, 0.7)
                self.nodes[i].add_connection(j, weight)
                self.nodes[j].add_connection(i, weight)
    
    def _build_small_world(self):
        """
        Build small-world network using Watts-Strogatz model.
        
        High clustering + short path lengths.
        """
        k = self.sw_k
        p = self.sw_p
        
        # Create ring lattice
        for i in range(self.num_nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % self.num_nodes
                weight = random.uniform(0.3, 0.7)
                self.nodes[i].add_connection(neighbor, weight)
                self.nodes[neighbor].add_connection(i, weight)
        
        # Rewire edges with probability p
        for i in range(self.num_nodes):
            neighbors = list(self.nodes[i].connections.keys())
            for j in neighbors:
                if random.random() < p:
                    # Rewire to random node
                    new_neighbor = random.randint(0, self.num_nodes - 1)
                    if new_neighbor != i and new_neighbor not in self.nodes[i].connections:
                        weight = self.nodes[i].connections[j]
                        self.nodes[i].remove_connection(j)
                        self.nodes[j].remove_connection(i)
                        self.nodes[i].add_connection(new_neighbor, weight)
                        self.nodes[new_neighbor].add_connection(i, weight)
    
    def _build_modular(self):
        """
        Build modular network.
        
        Dense intra-module connections, sparse inter-module connections.
        """
        # Intra-module connections (dense)
        for module in self.modules.values():
            nodes_list = list(module.nodes)
            
            for i, node_i in enumerate(nodes_list):
                for node_j in nodes_list[i + 1:]:
                    if random.random() < self.intra_density:
                        weight = random.uniform(0.4, 0.8)
                        self.nodes[node_i].add_connection(node_j, weight)
                        self.nodes[node_j].add_connection(node_i, weight)
        
        # Inter-module connections (sparse)
        for i, module_i in self.modules.items():
            for j, module_j in self.modules.items():
                if i >= j:
                    continue
                
                # Connect some nodes between modules
                nodes_i = list(module_i.nodes)
                nodes_j = list(module_j.nodes)
                
                num_connections = int(
                    len(nodes_i) * len(nodes_j) * self.inter_density
                )
                
                for _ in range(num_connections):
                    node_i = random.choice(nodes_i)
                    node_j = random.choice(nodes_j)
                    
                    if node_j not in self.nodes[node_i].connections:
                        weight = random.uniform(0.2, 0.5)
                        self.nodes[node_i].add_connection(node_j, weight)
                        self.nodes[node_j].add_connection(node_i, weight)
    
    def _build_hybrid(self):
        """
        Build hybrid network combining all three topologies.
        
        1. Start with modular structure
        2. Add scale-free hubs within modules
        3. Add small-world shortcuts
        """
        # Phase 1: Modular base
        self._build_modular()
        
        # Phase 2: Add scale-free hubs (preferential attachment within modules)
        for module in self.modules.values():
            nodes_list = list(module.nodes)
            if len(nodes_list) < 5:
                continue
            
            # Start with small complete graph
            m0 = min(3, len(nodes_list))
            for i in range(m0):
                for j in range(i + 1, m0):
                    node_i = nodes_list[i]
                    node_j = nodes_list[j]
                    if node_j not in self.nodes[node_i].connections:
                        weight = random.uniform(0.4, 0.7)
                        self.nodes[node_i].add_connection(node_j, weight)
                        self.nodes[node_j].add_connection(node_i, weight)
            
            # Preferential attachment within module
            for i in range(m0, len(nodes_list)):
                node_i = nodes_list[i]
                
                # Calculate degrees within module
                degrees = {}
                for j in range(i):
                    node_j = nodes_list[j]
                    # Count connections within module
                    module_degree = sum(
                        1 for conn in self.nodes[node_j].connections
                        if conn in module.nodes
                    )
                    degrees[node_j] = module_degree
                
                total_degree = sum(degrees.values())
                if total_degree == 0:
                    continue
                
                # Attach to 1-2 high-degree nodes
                probs = {j: deg / total_degree for j, deg in degrees.items()}
                num_attach = min(2, len(probs))
                
                for _ in range(num_attach):
                    remaining = [j for j in probs.keys() if j not in self.nodes[node_i].connections]
                    if not remaining:
                        break
                    target = random.choices(
                        remaining,
                        weights=[probs[j] for j in remaining]
                    )[0]
                    
                    weight = random.uniform(0.4, 0.7)
                    self.nodes[node_i].add_connection(target, weight)
                    self.nodes[target].add_connection(node_i, weight)
        
        # Phase 3: Add small-world shortcuts (rewire some edges)
        all_edges = []
        for node_id, node in self.nodes.items():
            for target_id in node.connections:
                if node_id < target_id:  # Avoid duplicates
                    all_edges.append((node_id, target_id))
        
        # Rewire 10% of edges randomly
        num_rewire = int(len(all_edges) * 0.1)
        edges_to_rewire = random.sample(all_edges, min(num_rewire, len(all_edges)))
        
        for node_i, node_j in edges_to_rewire:
            # Remove old edge
            weight = self.nodes[node_i].connections[node_j]
            self.nodes[node_i].remove_connection(node_j)
            self.nodes[node_j].remove_connection(node_i)
            
            # Add new random edge
            new_target = random.randint(0, self.num_nodes - 1)
            if new_target != node_i and new_target not in self.nodes[node_i].connections:
                self.nodes[node_i].add_connection(new_target, weight)
                self.nodes[new_target].add_connection(node_i, weight)
    
    def _identify_hubs(self, hub_threshold_percentile: float = 90):
        """Identify hub nodes (high degree)."""
        degrees = [node.degree for node in self.nodes.values()]
        threshold = np.percentile(degrees, hub_threshold_percentile)
        
        for node in self.nodes.values():
            if node.degree >= threshold:
                node.is_hub = True
                self.hub_nodes.add(node.node_id)
        
        self.stats['num_hubs'] = len(self.hub_nodes)
    
    def _compute_statistics(self):
        """Compute network statistics."""
        # Average degree
        degrees = [node.degree for node in self.nodes.values()]
        self.stats['avg_degree'] = np.mean(degrees)
        
        # Clustering coefficient
        clustering_coeffs = []
        for node in self.nodes.values():
            neighbors = list(node.connections.keys())
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count edges between neighbors
            edges_between = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1:]:
                    if n2 in self.nodes[n1].connections:
                        edges_between += 1
            
            # Clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            coeff = edges_between / possible_edges if possible_edges > 0 else 0.0
            clustering_coeffs.append(coeff)
            node.clustering_coeff = coeff
        
        self.stats['avg_clustering'] = np.mean(clustering_coeffs)
        
        # Average path length (sample-based for large networks)
        if self.num_nodes <= 100:
            self.stats['avg_path_length'] = self._compute_avg_path_length()
        else:
            self.stats['avg_path_length'] = self._estimate_avg_path_length(sample_size=100)
        
        # Modularity
        self.stats['modularity'] = self._compute_modularity()
        
        # Scale-free exponent (if applicable)
        if self.topology in [NetworkTopology.SCALE_FREE, NetworkTopology.HYBRID]:
            self.stats['scale_free_exponent'] = self._estimate_scale_free_exponent()
    
    def _compute_avg_path_length(self) -> float:
        """Compute average shortest path length (BFS)."""
        total_path_length = 0
        num_pairs = 0
        
        for source in range(min(50, self.num_nodes)):  # Sample for efficiency
            # BFS from source
            distances = {source: 0}
            queue = [source]
            
            while queue:
                current = queue.pop(0)
                current_dist = distances[current]
                
                for neighbor in self.nodes[current].connections:
                    if neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
            
            # Sum distances
            for dist in distances.values():
                if dist > 0:
                    total_path_length += dist
                    num_pairs += 1
        
        return total_path_length / num_pairs if num_pairs > 0 else 0.0
    
    def _estimate_avg_path_length(self, sample_size: int = 100) -> float:
        """Estimate average path length using sampling."""
        sampled_nodes = random.sample(range(self.num_nodes), min(sample_size, self.num_nodes))
        return self._compute_avg_path_length()
    
    def _compute_modularity(self) -> float:
        """
        Compute modularity Q.
        
        Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
        """
        m = sum(node.degree for node in self.nodes.values()) / 2  # Total edges
        
        if m == 0:
            return 0.0
        
        Q = 0.0
        for i, node_i in self.nodes.items():
            for j, node_j in self.nodes.items():
                # A_ij: 1 if connected, 0 otherwise
                A_ij = 1.0 if j in node_i.connections else 0.0
                
                # Expected edges
                expected = (node_i.degree * node_j.degree) / (2 * m)
                
                # δ(c_i, c_j): 1 if same module, 0 otherwise
                same_module = 1.0 if node_i.module_id == node_j.module_id else 0.0
                
                Q += (A_ij - expected) * same_module
        
        Q /= (2 * m)
        return Q
    
    def _estimate_scale_free_exponent(self) -> float:
        """
        Estimate scale-free exponent γ from degree distribution.
        
        P(k) ~ k^(-γ)
        """
        degrees = [node.degree for node in self.nodes.values() if node.degree > 0]
        
        if len(degrees) < 10:
            return 0.0
        
        # Log-log fit
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        log_degrees = []
        log_counts = []
        for degree, count in degree_counts.items():
            if degree > 0 and count > 0:
                log_degrees.append(math.log(degree))
                log_counts.append(math.log(count))
        
        if len(log_degrees) < 3:
            return 0.0
        
        # Linear regression
        n = len(log_degrees)
        sum_x = sum(log_degrees)
        sum_y = sum(log_counts)
        sum_xy = sum(x * y for x, y in zip(log_degrees, log_counts))
        sum_x2 = sum(x * x for x in log_degrees)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return -slope  # γ = -slope
    
    def get_node(self, node_id: int) -> Optional[NetworkNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_module(self, module_id: int) -> Optional[NetworkModule]:
        """Get module by ID."""
        return self.modules.get(module_id)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node."""
        node = self.nodes.get(node_id)
        return list(node.connections.keys()) if node else []
    
    def get_module_nodes(self, module_id: int) -> Set[int]:
        """Get all nodes in a module."""
        module = self.modules.get(module_id)
        return module.nodes if module else set()
    
    def get_hubs(self) -> List[int]:
        """Get list of hub node IDs."""
        return list(self.hub_nodes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            **self.stats,
            'num_nodes': self.num_nodes,
            'num_modules': self.num_modules,
            'topology': self.topology.value,
            'node_type': self.node_type,
        }
    
    def visualize_summary(self) -> str:
        """Get text summary of network structure."""
        lines = [
            f"Modular Network: {self.num_nodes} {self.node_type} nodes",
            f"Topology: {self.topology.value}",
            f"Modules: {self.num_modules}",
            f"",
            f"Statistics:",
            f"  Average degree: {self.stats['avg_degree']:.2f}",
            f"  Average clustering: {self.stats['avg_clustering']:.3f}",
            f"  Average path length: {self.stats['avg_path_length']:.2f}",
            f"  Modularity: {self.stats['modularity']:.3f}",
            f"  Hub nodes: {self.stats['num_hubs']} ({self.stats['num_hubs']/self.num_nodes*100:.1f}%)",
        ]
        
        if self.stats['scale_free_exponent'] > 0:
            lines.append(f"  Scale-free exponent: {self.stats['scale_free_exponent']:.2f}")
        
        return "\n".join(lines)
