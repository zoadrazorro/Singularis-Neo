"""
DATA-Brain Swarm Intelligence Layer

64+ micro-agents with scale-free topology, Hebbian dynamics, and emergent collective behavior.
Runs on AMD 6900XT (Router/Orchestrator) for distributed decision-making.

Architecture:
- 64 micro-agents organized in scale-free network
- Hebbian learning: "neurons that fire together, wire together"
- Emergent collective behavior through local interactions
- Integrates with ExpertArbiter for meta-cognitive routing
"""

from __future__ import annotations

import asyncio
import random
import time
import math
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from loguru import logger

try:
    from ..core.modular_network import ModularNetwork, NetworkTopology, ModuleType
    MODULAR_NETWORK_AVAILABLE = True
except ImportError:
    MODULAR_NETWORK_AVAILABLE = False
    logger.warning("[SWARM] Modular network not available")


class AgentRole(Enum):
    """Micro-agent specialization roles."""
    PERCEPTION = "perception"          # Sensory processing
    MEMORY = "memory"                  # Memory retrieval
    REASONING = "reasoning"            # Logical reasoning
    EMOTION = "emotion"                # Emotional processing
    ACTION = "action"                  # Action planning
    COORDINATION = "coordination"      # Inter-agent coordination
    LEARNING = "learning"              # Learning and adaptation
    SYNTHESIS = "synthesis"            # Information synthesis


@dataclass
class MicroAgent:
    """Single micro-agent in the swarm."""
    agent_id: int
    role: AgentRole
    activation: float = 0.0            # Current activation level [0, 1]
    connections: Dict[int, float] = field(default_factory=dict)  # agent_id -> weight
    memory: List[Any] = field(default_factory=list)  # Short-term memory
    specialization_score: float = 1.0  # How specialized this agent is
    
    def activate(self, input_signal: float, decay: float = 0.95):
        """Update activation based on input signal."""
        self.activation = min(1.0, self.activation * decay + input_signal)
    
    def hebbian_update(self, other_id: int, correlation: float, learning_rate: float = 0.01):
        """
        Hebbian learning: strengthen connections between co-active agents.
        
        "Neurons that fire together, wire together"
        """
        if other_id not in self.connections:
            self.connections[other_id] = 0.0
        
        # Hebbian rule: Δw = η * x_i * x_j
        delta_w = learning_rate * correlation
        self.connections[other_id] = np.clip(
            self.connections[other_id] + delta_w,
            -1.0,  # Inhibitory connections
            1.0    # Excitatory connections
        )


@dataclass
class SwarmState:
    """Current state of the swarm."""
    cycle: int = 0
    total_activation: float = 0.0
    coherence: float = 0.0             # How synchronized the swarm is
    emergent_patterns: List[str] = field(default_factory=list)
    decision_confidence: float = 0.0


class SwarmIntelligence:
    """
    Swarm intelligence layer with 64+ micro-agents.
    
    Features:
    - Scale-free topology (power-law degree distribution)
    - Hebbian dynamics (connection strengthening)
    - Emergent collective behavior
    - Distributed decision-making
    """
    
    def __init__(
        self,
        num_agents: int = 64,
        topology: str = "scale_free",  # "scale_free", "small_world", "random"
        hebbian_learning_rate: float = 0.01,
        activation_decay: float = 0.95,
    ):
        """
        Initialize swarm intelligence layer.
        
        Args:
            num_agents: Number of micro-agents (default 64)
            topology: Network topology type
            hebbian_learning_rate: Learning rate for Hebbian updates
            activation_decay: Decay rate for agent activation
        """
        self.num_agents = num_agents
        self.topology_type = topology
        self.hebbian_lr = hebbian_learning_rate
        self.activation_decay = activation_decay
        
        # Create micro-agents
        self.agents: Dict[int, MicroAgent] = {}
        self._initialize_agents()
        
        # Build network topology using ModularNetwork
        self.modular_network: Optional[ModularNetwork] = None
        if MODULAR_NETWORK_AVAILABLE:
            self._build_modular_topology()
        else:
            # Fallback to manual topology
            self._build_topology()
        
        # Swarm state
        self.state = SwarmState()
        
        # Statistics
        self.stats = {
            'total_cycles': 0,
            'decisions_made': 0,
            'emergent_patterns_detected': 0,
            'avg_coherence': 0.0,
            'hebbian_updates': 0,
        }
        
        logger.info(
            f"[SWARM] Initialized {num_agents} micro-agents with {topology} topology"
        )
    
    def _initialize_agents(self):
        """Initialize micro-agents with role distribution."""
        roles = list(AgentRole)
        
        for i in range(self.num_agents):
            # Assign roles (roughly balanced)
            role = roles[i % len(roles)]
            
            self.agents[i] = MicroAgent(
                agent_id=i,
                role=role,
                activation=random.uniform(0.0, 0.1),  # Small initial activation
            )
    
    def _build_modular_topology(self):
        """Build topology using ModularNetwork (brain-like)."""
        # Map topology type
        topology_map = {
            "scale_free": NetworkTopology.SCALE_FREE,
            "small_world": NetworkTopology.SMALL_WORLD,
            "random": NetworkTopology.MODULAR,
        }
        
        network_topology = topology_map.get(self.topology_type, NetworkTopology.HYBRID)
        
        # Create modular network
        num_modules = 8  # One per AgentRole
        self.modular_network = ModularNetwork(
            num_nodes=self.num_agents,
            num_modules=num_modules,
            topology=network_topology,
            node_type="swarm_agent",
            intra_module_density=0.3,
            inter_module_density=0.05,
        )
        
        # Copy connections from modular network to agents
        for agent_id, agent in self.agents.items():
            network_node = self.modular_network.get_node(agent_id)
            if network_node:
                agent.connections = network_node.connections.copy()
        
        logger.info(
            f"[SWARM] Built modular topology: "
            f"{self.modular_network.stats['avg_degree']:.1f} avg degree, "
            f"{self.modular_network.stats['avg_clustering']:.3f} clustering"
        )
    
    def _build_topology(self):
        """Build network topology between agents (fallback)."""
        if self.topology_type == "scale_free":
            self._build_scale_free_topology()
        elif self.topology_type == "small_world":
            self._build_small_world_topology()
        else:
            self._build_random_topology()
    
    def _build_scale_free_topology(self):
        """
        Build scale-free network using preferential attachment.
        
        Power-law degree distribution: P(k) ~ k^(-γ)
        Few highly connected hubs, many sparsely connected nodes.
        """
        # Start with small complete graph
        m0 = 3  # Initial nodes
        m = 2   # Edges to attach per new node
        
        # Initialize with complete graph
        for i in range(m0):
            for j in range(i + 1, m0):
                weight = random.uniform(0.1, 0.5)
                self.agents[i].connections[j] = weight
                self.agents[j].connections[i] = weight
        
        # Preferential attachment
        for i in range(m0, self.num_agents):
            # Calculate connection probabilities (proportional to degree)
            degrees = {
                j: len(self.agents[j].connections)
                for j in range(i)
            }
            total_degree = sum(degrees.values())
            
            if total_degree == 0:
                # Fallback: random connections
                targets = random.sample(range(i), min(m, i))
            else:
                # Preferential attachment
                probs = {j: deg / total_degree for j, deg in degrees.items()}
                targets = []
                for _ in range(min(m, i)):
                    # Sample without replacement
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
                weight = random.uniform(0.1, 0.5)
                self.agents[i].connections[j] = weight
                self.agents[j].connections[i] = weight
        
        logger.info(f"[SWARM] Built scale-free topology with {self._count_edges()} edges")
    
    def _build_small_world_topology(self):
        """Build small-world network (Watts-Strogatz model)."""
        k = 4  # Each node connected to k nearest neighbors
        p = 0.1  # Rewiring probability
        
        # Create ring lattice
        for i in range(self.num_agents):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % self.num_agents
                weight = random.uniform(0.1, 0.5)
                self.agents[i].connections[neighbor] = weight
                self.agents[neighbor].connections[i] = weight
        
        # Rewire edges with probability p
        for i in range(self.num_agents):
            neighbors = list(self.agents[i].connections.keys())
            for j in neighbors:
                if random.random() < p:
                    # Rewire to random node
                    new_neighbor = random.randint(0, self.num_agents - 1)
                    if new_neighbor != i and new_neighbor not in self.agents[i].connections:
                        weight = self.agents[i].connections[j]
                        del self.agents[i].connections[j]
                        del self.agents[j].connections[i]
                        self.agents[i].connections[new_neighbor] = weight
                        self.agents[new_neighbor].connections[i] = weight
        
        logger.info(f"[SWARM] Built small-world topology with {self._count_edges()} edges")
    
    def _build_random_topology(self):
        """Build random network (Erdős-Rényi model)."""
        p = 0.1  # Connection probability
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if random.random() < p:
                    weight = random.uniform(0.1, 0.5)
                    self.agents[i].connections[j] = weight
                    self.agents[j].connections[i] = weight
        
        logger.info(f"[SWARM] Built random topology with {self._count_edges()} edges")
    
    def _count_edges(self) -> int:
        """Count total edges in network."""
        return sum(len(agent.connections) for agent in self.agents.values()) // 2
    
    async def process_query(
        self,
        query: str,
        context: Dict[str, Any],
        expert_selection: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Process query through swarm intelligence.
        
        Args:
            query: Input query
            context: Query context
            expert_selection: Optional pre-selected experts
            
        Returns:
            Swarm decision with confidence and emergent patterns
        """
        self.state.cycle += 1
        self.stats['total_cycles'] += 1
        
        logger.info(f"[SWARM] Processing query (cycle {self.state.cycle})")
        
        # Phase 1: Activate agents based on query
        await self._activate_agents(query, context)
        
        # Phase 2: Propagate activation through network
        await self._propagate_activation()
        
        # Phase 3: Hebbian learning (strengthen co-active connections)
        self._hebbian_learning()
        
        # Phase 4: Detect emergent patterns
        patterns = self._detect_emergent_patterns()
        
        # Phase 5: Make collective decision
        decision = self._collective_decision(expert_selection)
        
        # Update state
        self.state.emergent_patterns = patterns
        self.state.decision_confidence = decision['confidence']
        
        self.stats['decisions_made'] += 1
        if patterns:
            self.stats['emergent_patterns_detected'] += len(patterns)
        
        return decision
    
    async def _activate_agents(self, query: str, context: Dict[str, Any]):
        """Activate agents based on query content."""
        query_lower = query.lower()
        
        for agent in self.agents.values():
            # Role-based activation
            activation = 0.0
            
            if agent.role == AgentRole.PERCEPTION:
                if any(kw in query_lower for kw in ['see', 'image', 'visual', 'perceive']):
                    activation = 0.8
            
            elif agent.role == AgentRole.MEMORY:
                if any(kw in query_lower for kw in ['remember', 'recall', 'past', 'history']):
                    activation = 0.8
            
            elif agent.role == AgentRole.REASONING:
                if any(kw in query_lower for kw in ['why', 'reason', 'logic', 'because']):
                    activation = 0.8
            
            elif agent.role == AgentRole.EMOTION:
                if any(kw in query_lower for kw in ['feel', 'emotion', 'mood', 'sentiment']):
                    activation = 0.8
            
            elif agent.role == AgentRole.ACTION:
                if any(kw in query_lower for kw in ['should', 'do', 'action', 'plan']):
                    activation = 0.8
            
            elif agent.role == AgentRole.LEARNING:
                if any(kw in query_lower for kw in ['learn', 'pattern', 'trend']):
                    activation = 0.7
            
            elif agent.role == AgentRole.SYNTHESIS:
                activation = 0.5  # Always somewhat active
            
            # Context-based activation
            if 'life' in str(context).lower():
                if agent.role in [AgentRole.MEMORY, AgentRole.EMOTION]:
                    activation += 0.2
            
            agent.activate(activation, decay=self.activation_decay)
    
    async def _propagate_activation(self, iterations: int = 3):
        """Propagate activation through network."""
        for _ in range(iterations):
            # Calculate new activations
            new_activations = {}
            
            for agent_id, agent in self.agents.items():
                # Sum weighted inputs from connected agents
                input_sum = 0.0
                for neighbor_id, weight in agent.connections.items():
                    neighbor = self.agents[neighbor_id]
                    input_sum += weight * neighbor.activation
                
                # Apply sigmoid activation function
                new_activation = 1.0 / (1.0 + math.exp(-input_sum))
                new_activations[agent_id] = new_activation
            
            # Update activations
            for agent_id, new_activation in new_activations.items():
                self.agents[agent_id].activation = new_activation
            
            await asyncio.sleep(0)  # Yield control
    
    def _hebbian_learning(self):
        """Apply Hebbian learning to strengthen co-active connections."""
        # Find highly active agents
        active_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.activation > 0.5
        ]
        
        # Update connections between co-active agents
        for i, agent_id_i in enumerate(active_agents):
            agent_i = self.agents[agent_id_i]
            
            for agent_id_j in active_agents[i + 1:]:
                agent_j = self.agents[agent_id_j]
                
                # Correlation = product of activations
                correlation = agent_i.activation * agent_j.activation
                
                # Hebbian update
                agent_i.hebbian_update(agent_id_j, correlation, self.hebbian_lr)
                agent_j.hebbian_update(agent_id_i, correlation, self.hebbian_lr)
                
                self.stats['hebbian_updates'] += 1
    
    def _detect_emergent_patterns(self) -> List[str]:
        """Detect emergent patterns in swarm activation."""
        patterns = []
        
        # Pattern 1: Role clustering (agents of same role highly active)
        role_activations = {}
        for agent in self.agents.values():
            if agent.role not in role_activations:
                role_activations[agent.role] = []
            role_activations[agent.role].append(agent.activation)
        
        for role, activations in role_activations.items():
            avg_activation = np.mean(activations)
            if avg_activation > 0.7:
                patterns.append(f"high_{role.value}_activity")
        
        # Pattern 2: Synchronization (many agents with similar activation)
        all_activations = [agent.activation for agent in self.agents.values()]
        activation_std = np.std(all_activations)
        
        if activation_std < 0.1:
            patterns.append("synchronized_swarm")
        elif activation_std > 0.4:
            patterns.append("diverse_activation")
        
        # Pattern 3: Hub activation (highly connected agents active)
        hub_agents = sorted(
            self.agents.values(),
            key=lambda a: len(a.connections),
            reverse=True
        )[:5]
        
        hub_activation = np.mean([a.activation for a in hub_agents])
        if hub_activation > 0.7:
            patterns.append("hub_coordination")
        
        # Update coherence
        self.state.coherence = 1.0 - activation_std
        self.stats['avg_coherence'] = (
            (self.stats['avg_coherence'] * (self.stats['total_cycles'] - 1) + self.state.coherence) /
            self.stats['total_cycles']
        )
        
        return patterns
    
    def _collective_decision(self, expert_selection: Optional[Set[str]]) -> Dict[str, Any]:
        """Make collective decision based on swarm state."""
        # Aggregate activations by role
        role_votes = {}
        for agent in self.agents.values():
            if agent.role not in role_votes:
                role_votes[agent.role] = []
            role_votes[agent.role].append(agent.activation)
        
        # Calculate role consensus
        role_consensus = {
            role: np.mean(activations)
            for role, activations in role_votes.items()
        }
        
        # Select top roles
        top_roles = sorted(
            role_consensus.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Calculate confidence
        confidence = np.mean([score for _, score in top_roles])
        
        # Map roles to expert domains
        role_to_expert = {
            AgentRole.PERCEPTION: "vision",
            AgentRole.MEMORY: "memory",
            AgentRole.REASONING: "reasoning",
            AgentRole.EMOTION: "emotion",
            AgentRole.ACTION: "action",
            AgentRole.LEARNING: "analysis",
            AgentRole.SYNTHESIS: "synthesis",
            AgentRole.COORDINATION: "planning",
        }
        
        recommended_experts = set()
        for role, score in top_roles:
            if role in role_to_expert:
                recommended_experts.add(role_to_expert[role])
        
        # Always include synthesis
        recommended_experts.add("synthesis")
        
        return {
            'recommended_experts': recommended_experts,
            'confidence': confidence,
            'role_consensus': {role.value: score for role, score in role_consensus.items()},
            'emergent_patterns': self.state.emergent_patterns,
            'swarm_coherence': self.state.coherence,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get swarm statistics."""
        return {
            **self.stats,
            'num_agents': self.num_agents,
            'num_edges': self._count_edges(),
            'current_coherence': self.state.coherence,
            'current_cycle': self.state.cycle,
        }
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network topology metrics."""
        # Degree distribution
        degrees = [len(agent.connections) for agent in self.agents.values()]
        
        # Clustering coefficient (local)
        clustering_coeffs = []
        for agent in self.agents.values():
            neighbors = list(agent.connections.keys())
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count edges between neighbors
            edges_between = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1:]:
                    if n2 in self.agents[n1].connections:
                        edges_between += 1
            
            # Clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeffs.append(edges_between / possible_edges if possible_edges > 0 else 0.0)
        
        return {
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'degree_std': np.std(degrees),
            'avg_clustering': np.mean(clustering_coeffs),
            'topology_type': self.topology_type,
        }
