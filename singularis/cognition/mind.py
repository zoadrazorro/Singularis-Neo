"""
Modular Mind System - Unified Cognitive Architecture

Integrates:
1. Theory of Mind (ToM) - Understanding mental states of self and others
2. Heuristic Differential Analysis - Fast pattern-based reasoning
3. Multi-Node Cross-Parallelism - Distributed world-domain processing
4. Cognitive Coherence Valence - Measuring belief consistency
5. Cognitive Dissonance Detection - Identifying contradictions
6. Web Graph Network - Interconnected cognitive nodes

This module serves as the central cognitive hub for the entire AGI system.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from loguru import logger


class MentalState(Enum):
    """Mental states for Theory of Mind."""
    BELIEF = "belief"
    DESIRE = "desire"
    INTENTION = "intention"
    KNOWLEDGE = "knowledge"
    PERCEPTION = "perception"
    EMOTION = "emotion"
    EXPECTATION = "expectation"


class CognitiveValence(Enum):
    """Valence of cognitive states."""
    POSITIVE = "positive"  # Coherent, aligned
    NEUTRAL = "neutral"    # Uncertain, ambiguous
    NEGATIVE = "negative"  # Dissonant, conflicting


@dataclass
class MentalStateRepresentation:
    """Representation of a mental state (self or other)."""
    agent: str  # "self" or NPC name
    state_type: MentalState
    content: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)


@dataclass
class HeuristicPattern:
    """Fast heuristic pattern for decision-making."""
    pattern_id: str
    condition: str  # Pattern condition
    action: str     # Recommended action
    success_rate: float
    usage_count: int = 0
    last_used: float = 0.0


@dataclass
class CognitiveNode:
    """Node in the cognitive web graph."""
    node_id: str
    domain: str  # world, combat, social, navigation, etc.
    beliefs: Dict[str, float]  # belief -> confidence
    connections: Set[str]  # Connected node IDs
    activation_level: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class CoherenceCheck:
    """Result of cognitive coherence analysis."""
    coherence_score: float  # 0-1, higher is more coherent
    valence: CognitiveValence
    dissonances: List[Tuple[str, str, float]]  # (belief1, belief2, conflict_strength)
    recommendations: List[str]


class TheoryOfMind:
    """
    Theory of Mind module - understanding mental states.
    
    Tracks:
    - Self mental states (beliefs, desires, intentions)
    - Other agents' mental states (NPCs, enemies)
    - Perspective taking
    - False belief understanding
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Mental state tracking
        self.self_states: Dict[MentalState, List[MentalStateRepresentation]] = {
            state: [] for state in MentalState
        }
        
        self.other_states: Dict[str, Dict[MentalState, List[MentalStateRepresentation]]] = {}
        
        # Perspective tracking
        self.perspective_history: List[Tuple[str, str, float]] = []  # (agent, perspective, timestamp)
        
        if verbose:
            print("[ToM] Theory of Mind initialized")
    
    def update_self_state(
        self,
        state_type: MentalState,
        content: str,
        confidence: float,
        evidence: Optional[List[str]] = None
    ):
        """Update self mental state."""
        state = MentalStateRepresentation(
            agent="self",
            state_type=state_type,
            content=content,
            confidence=confidence,
            evidence=evidence or []
        )
        
        self.self_states[state_type].append(state)
        
        # Keep only recent states (last 100 per type)
        if len(self.self_states[state_type]) > 100:
            self.self_states[state_type].pop(0)
    
    def infer_other_state(
        self,
        agent: str,
        state_type: MentalState,
        content: str,
        confidence: float,
        evidence: List[str]
    ):
        """Infer another agent's mental state."""
        if agent not in self.other_states:
            self.other_states[agent] = {state: [] for state in MentalState}
        
        state = MentalStateRepresentation(
            agent=agent,
            state_type=state_type,
            content=content,
            confidence=confidence,
            evidence=evidence
        )
        
        self.other_states[agent][state_type].append(state)
        
        if len(self.other_states[agent][state_type]) > 50:
            self.other_states[agent][state_type].pop(0)
    
    def take_perspective(self, agent: str) -> Dict[MentalState, str]:
        """Take another agent's perspective."""
        if agent not in self.other_states:
            return {}
        
        perspective = {}
        for state_type, states in self.other_states[agent].items():
            if states:
                # Get most recent state
                latest = states[-1]
                perspective[state_type] = latest.content
        
        self.perspective_history.append((agent, str(perspective), time.time()))
        
        return perspective
    
    def predict_behavior(self, agent: str, situation: Dict[str, Any]) -> str:
        """Predict agent's behavior based on their mental states."""
        if agent not in self.other_states:
            return "Unknown - insufficient mental state information"
        
        # Get beliefs, desires, and intentions
        beliefs = self.other_states[agent][MentalState.BELIEF]
        desires = self.other_states[agent][MentalState.DESIRE]
        intentions = self.other_states[agent][MentalState.INTENTION]
        
        if not (beliefs or desires or intentions):
            return "Unknown - no mental states tracked"
        
        # Simple BDI (Belief-Desire-Intention) prediction
        prediction = f"{agent} likely to: "
        
        if intentions:
            prediction += intentions[-1].content
        elif desires:
            prediction += f"pursue {desires[-1].content}"
        elif beliefs:
            prediction += f"act based on belief: {beliefs[-1].content}"
        
        return prediction


class HeuristicDifferentialAnalyzer:
    """
    Heuristic Differential Analysis - fast pattern-based reasoning.
    
    Uses learned heuristics for quick decision-making:
    - Pattern matching
    - Fast approximations
    - Differential analysis (what changed?)
    - Similarity-based reasoning
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Heuristic patterns
        self.patterns: Dict[str, HeuristicPattern] = {}
        
        # Differential state tracking
        self.last_state: Optional[Dict[str, Any]] = None
        self.state_deltas: deque = deque(maxlen=100)
        
        if verbose:
            print("[HDA] Heuristic Differential Analyzer initialized")
    
    def add_pattern(
        self,
        pattern_id: str,
        condition: str,
        action: str,
        initial_success_rate: float = 0.5
    ):
        """Add a heuristic pattern."""
        self.patterns[pattern_id] = HeuristicPattern(
            pattern_id=pattern_id,
            condition=condition,
            action=action,
            success_rate=initial_success_rate
        )
    
    def match_pattern(self, current_state: Dict[str, Any]) -> Optional[HeuristicPattern]:
        """Match current state to heuristic patterns."""
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            # Simple pattern matching (in production, use more sophisticated matching)
            score = self._calculate_match_score(pattern, current_state)
            
            if score > best_score and score > 0.6:  # Threshold
                best_score = score
                best_match = pattern
        
        if best_match:
            best_match.usage_count += 1
            best_match.last_used = time.time()
        
        return best_match
    
    def _calculate_match_score(
        self,
        pattern: HeuristicPattern,
        state: Dict[str, Any]
    ) -> float:
        """Calculate how well a pattern matches current state."""
        # Simplified matching - check if key terms in condition appear in state
        condition_terms = set(pattern.condition.lower().split())
        state_terms = set(str(state).lower().split())
        
        overlap = len(condition_terms & state_terms)
        total = len(condition_terms)
        
        return overlap / total if total > 0 else 0.0
    
    def analyze_differential(
        self,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze what changed from last state (differential analysis)."""
        if self.last_state is None:
            self.last_state = current_state
            return {'changes': {}, 'magnitude': 0.0}
        
        changes = {}
        total_change = 0.0
        
        # Find what changed
        all_keys = set(self.last_state.keys()) | set(current_state.keys())
        
        for key in all_keys:
            old_val = self.last_state.get(key)
            new_val = current_state.get(key)
            
            if old_val != new_val:
                changes[key] = {
                    'old': old_val,
                    'new': new_val,
                    'delta': self._calculate_delta(old_val, new_val)
                }
                total_change += abs(changes[key]['delta'])
        
        differential = {
            'changes': changes,
            'magnitude': total_change,
            'timestamp': time.time()
        }
        
        self.state_deltas.append(differential)
        self.last_state = current_state.copy()
        
        return differential
    
    def _calculate_delta(self, old_val: Any, new_val: Any) -> float:
        """Calculate delta between values."""
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            return new_val - old_val
        elif old_val == new_val:
            return 0.0
        else:
            return 1.0  # Categorical change
    
    def update_pattern_success(self, pattern_id: str, success: bool):
        """Update pattern success rate based on outcome."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            # Exponential moving average
            alpha = 0.1
            new_success = 1.0 if success else 0.0
            pattern.success_rate = (alpha * new_success + 
                                   (1 - alpha) * pattern.success_rate)


class MultiNodeCrossParallelism:
    """
    Multi-Node Cross-Parallelism - distributed world-domain processing.
    
    Features:
    - Multiple cognitive nodes for different domains
    - Parallel processing across nodes
    - Cross-domain information flow
    - Emergent global understanding
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Cognitive nodes by domain
        self.nodes: Dict[str, CognitiveNode] = {}
        
        # Cross-domain connections (web graph)
        self.connection_strengths: Dict[Tuple[str, str], float] = {}
        
        # Global activation pattern
        self.global_activation: Dict[str, float] = {}
        
        if verbose:
            print("[MNCP] Multi-Node Cross-Parallelism initialized")
    
    def create_node(
        self,
        node_id: str,
        domain: str,
        initial_beliefs: Optional[Dict[str, float]] = None
    ):
        """Create a cognitive node."""
        self.nodes[node_id] = CognitiveNode(
            node_id=node_id,
            domain=domain,
            beliefs=initial_beliefs or {},
            connections=set()
        )
        
        if self.verbose:
            print(f"[MNCP] Created node: {node_id} (domain: {domain})")
    
    def connect_nodes(self, node1_id: str, node2_id: str, strength: float = 0.5):
        """Connect two nodes in the web graph."""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id].connections.add(node2_id)
            self.nodes[node2_id].connections.add(node1_id)
            
            self.connection_strengths[(node1_id, node2_id)] = strength
            self.connection_strengths[(node2_id, node1_id)] = strength
    
    def activate_node(self, node_id: str, activation: float):
        """Activate a node (set activation level)."""
        if node_id in self.nodes:
            self.nodes[node_id].activation_level = activation
            self.nodes[node_id].last_updated = time.time()
    
    def propagate_activation(self, iterations: int = 3):
        """Propagate activation through the web graph."""
        for _ in range(iterations):
            new_activations = {}
            
            for node_id, node in self.nodes.items():
                # Calculate incoming activation from connected nodes
                incoming = 0.0
                
                for connected_id in node.connections:
                    if connected_id in self.nodes:
                        strength = self.connection_strengths.get(
                            (connected_id, node_id), 0.5
                        )
                        incoming += (self.nodes[connected_id].activation_level * 
                                   strength)
                
                # Combine current and incoming activation
                new_activation = 0.7 * node.activation_level + 0.3 * incoming
                new_activations[node_id] = np.clip(new_activation, 0.0, 1.0)
            
            # Update all activations
            for node_id, activation in new_activations.items():
                self.nodes[node_id].activation_level = activation
        
        # Update global activation pattern
        self.global_activation = {
            node_id: node.activation_level
            for node_id, node in self.nodes.items()
        }
    
    async def parallel_process(
        self,
        processing_fn: callable,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process inputs in parallel across nodes."""
        tasks = []
        
        for node_id, node in self.nodes.items():
            # Create task for each node
            task = asyncio.create_task(
                processing_fn(node, inputs.get(node.domain, {}))
            )
            tasks.append((node_id, task))
        
        # Wait for all tasks
        results = {}
        for node_id, task in tasks:
            try:
                result = await task
                results[node_id] = result
            except Exception as e:
                logger.error(f"[MNCP] Node {node_id} processing failed: {e}")
                results[node_id] = None
        
        return results
    
    def get_cross_domain_insights(self) -> List[str]:
        """Extract insights from cross-domain activation patterns."""
        insights = []
        
        # Find highly activated nodes
        active_nodes = [
            (node_id, node)
            for node_id, node in self.nodes.items()
            if node.activation_level > 0.7
        ]
        
        if len(active_nodes) >= 2:
            # Multiple domains highly activated - cross-domain pattern
            domains = [node.domain for _, node in active_nodes]
            insights.append(
                f"Cross-domain activation: {', '.join(domains)}"
            )
        
        # Find strong connections between different domains
        for (node1_id, node2_id), strength in self.connection_strengths.items():
            if strength > 0.8:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                if node1.domain != node2.domain:
                    insights.append(
                        f"Strong cross-domain link: {node1.domain} ↔ {node2.domain}"
                    )
        
        return insights


class CognitiveCoherenceAnalyzer:
    """
    Cognitive Coherence and Dissonance Analysis.
    
    Measures:
    - Belief consistency (coherence)
    - Contradictions (dissonance)
    - Cognitive valence (positive/neutral/negative)
    - Resolution recommendations
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Belief network
        self.beliefs: Dict[str, float] = {}  # belief -> confidence
        
        # Belief relationships
        self.supports: Dict[str, Set[str]] = defaultdict(set)  # belief -> supporting beliefs
        self.contradicts: Dict[str, Set[str]] = defaultdict(set)  # belief -> contradicting beliefs
        
        # Coherence history
        self.coherence_history: deque = deque(maxlen=100)
        
        if verbose:
            print("[CCA] Cognitive Coherence Analyzer initialized")
    
    def add_belief(self, belief: str, confidence: float):
        """Add or update a belief."""
        self.beliefs[belief] = confidence
    
    def add_support(self, belief1: str, belief2: str):
        """Mark that belief1 supports belief2."""
        self.supports[belief2].add(belief1)
    
    def add_contradiction(self, belief1: str, belief2: str):
        """Mark that belief1 contradicts belief2."""
        self.contradicts[belief1].add(belief2)
        self.contradicts[belief2].add(belief1)
    
    def check_coherence(self) -> CoherenceCheck:
        """Check cognitive coherence and detect dissonance."""
        if not self.beliefs:
            return CoherenceCheck(
                coherence_score=1.0,
                valence=CognitiveValence.NEUTRAL,
                dissonances=[],
                recommendations=[]
            )
        
        # Find contradictions
        dissonances = []
        
        for belief1, contradicting in self.contradicts.items():
            if belief1 in self.beliefs:
                conf1 = self.beliefs[belief1]
                
                for belief2 in contradicting:
                    if belief2 in self.beliefs:
                        conf2 = self.beliefs[belief2]
                        
                        # Conflict strength = product of confidences
                        conflict_strength = conf1 * conf2
                        
                        if conflict_strength > 0.3:  # Significant conflict
                            dissonances.append((belief1, belief2, conflict_strength))
        
        # Calculate coherence score
        total_possible_conflicts = len(self.beliefs) * (len(self.beliefs) - 1) / 2
        actual_conflicts = len(dissonances)
        
        coherence_score = 1.0 - (actual_conflicts / max(total_possible_conflicts, 1))
        
        # Determine valence
        if coherence_score > 0.8:
            valence = CognitiveValence.POSITIVE
        elif coherence_score > 0.5:
            valence = CognitiveValence.NEUTRAL
        else:
            valence = CognitiveValence.NEGATIVE
        
        # Generate recommendations
        recommendations = []
        
        for belief1, belief2, strength in dissonances:
            if strength > 0.5:
                # Strong dissonance - recommend resolution
                conf1 = self.beliefs[belief1]
                conf2 = self.beliefs[belief2]
                
                if conf1 > conf2:
                    recommendations.append(
                        f"Consider revising: '{belief2}' (conflicts with higher-confidence '{belief1}')"
                    )
                else:
                    recommendations.append(
                        f"Consider revising: '{belief1}' (conflicts with higher-confidence '{belief2}')"
                    )
        
        result = CoherenceCheck(
            coherence_score=coherence_score,
            valence=valence,
            dissonances=dissonances,
            recommendations=recommendations
        )
        
        self.coherence_history.append((coherence_score, time.time()))
        
        return result
    
    def resolve_dissonance(
        self,
        belief1: str,
        belief2: str,
        resolution_strategy: str = "confidence"
    ):
        """Resolve cognitive dissonance between two beliefs."""
        if belief1 not in self.beliefs or belief2 not in self.beliefs:
            return
        
        if resolution_strategy == "confidence":
            # Keep higher confidence belief
            if self.beliefs[belief1] > self.beliefs[belief2]:
                del self.beliefs[belief2]
                if self.verbose:
                    print(f"[CCA] Resolved dissonance: removed '{belief2}'")
            else:
                del self.beliefs[belief1]
                if self.verbose:
                    print(f"[CCA] Resolved dissonance: removed '{belief1}'")
        
        elif resolution_strategy == "integrate":
            # Try to integrate both beliefs
            avg_confidence = (self.beliefs[belief1] + self.beliefs[belief2]) / 2
            integrated = f"({belief1}) AND ({belief2})"
            
            del self.beliefs[belief1]
            del self.beliefs[belief2]
            self.beliefs[integrated] = avg_confidence
            
            if self.verbose:
                print(f"[CCA] Integrated beliefs: '{integrated}'")


class Mind:
    """
    Unified Mind System - integrates all cognitive modules.
    
    Combines:
    - Theory of Mind
    - Heuristic Differential Analysis
    - Multi-Node Cross-Parallelism
    - Cognitive Coherence Analysis
    
    Organized as a web graph network connecting all components.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Core cognitive modules
        self.theory_of_mind = TheoryOfMind(verbose=verbose)
        self.heuristic_analyzer = HeuristicDifferentialAnalyzer(verbose=verbose)
        self.multi_node = MultiNodeCrossParallelism(verbose=verbose)
        self.coherence_analyzer = CognitiveCoherenceAnalyzer(verbose=verbose)
        
        # Initialize default cognitive nodes
        self._initialize_default_nodes()
        
        # Mind state
        self.mind_state = {
            'coherence': 1.0,
            'dissonance_level': 0.0,
            'active_domains': set(),
            'perspective': 'self'
        }
        
        if verbose:
            print("\n" + "="*80)
            print("MIND SYSTEM INITIALIZED".center(80))
            print("="*80)
            print("[OK] Theory of Mind")
            print("[OK] Heuristic Differential Analyzer")
            print("[OK] Multi-Node Cross-Parallelism")
            print("[OK] Cognitive Coherence Analyzer")
            print("="*80 + "\n")
    
    def _initialize_default_nodes(self):
        """Initialize default cognitive nodes for common domains."""
        domains = [
            'world_model',
            'combat',
            'social',
            'navigation',
            'resource_management',
            'self_awareness',
            'other_awareness'
        ]
        
        for domain in domains:
            self.multi_node.create_node(
                node_id=f"{domain}_node",
                domain=domain
            )
        
        # Create cross-domain connections
        connections = [
            ('world_model_node', 'combat_node', 0.8),
            ('world_model_node', 'social_node', 0.7),
            ('world_model_node', 'navigation_node', 0.9),
            ('combat_node', 'resource_management_node', 0.6),
            ('social_node', 'other_awareness_node', 0.9),
            ('self_awareness_node', 'other_awareness_node', 0.7),
        ]
        
        for node1, node2, strength in connections:
            self.multi_node.connect_nodes(node1, node2, strength)
    
    async def process_situation(
        self,
        situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a situation through the entire mind system.
        
        Returns comprehensive cognitive analysis.
        """
        # 1. Theory of Mind - update self state
        self.theory_of_mind.update_self_state(
            state_type=MentalState.PERCEPTION,
            content=str(situation),
            confidence=0.9,
            evidence=[f"Direct observation at {time.time()}"]
        )
        
        # 2. Heuristic Analysis - find patterns and differentials
        matched_pattern = self.heuristic_analyzer.match_pattern(situation)
        differential = self.heuristic_analyzer.analyze_differential(situation)
        
        # 3. Multi-Node Processing - activate relevant domains
        for domain in situation.get('active_domains', []):
            node_id = f"{domain}_node"
            if node_id in self.multi_node.nodes:
                self.multi_node.activate_node(node_id, 0.8)
        
        # Propagate activation through web graph
        self.multi_node.propagate_activation(iterations=3)
        
        # Get cross-domain insights
        cross_insights = self.multi_node.get_cross_domain_insights()
        
        # 4. Coherence Check
        coherence_check = self.coherence_analyzer.check_coherence()
        
        # Update mind state
        self.mind_state['coherence'] = coherence_check.coherence_score
        self.mind_state['dissonance_level'] = len(coherence_check.dissonances) / max(len(self.coherence_analyzer.beliefs), 1)
        self.mind_state['active_domains'] = set(
            node_id for node_id, activation in self.multi_node.global_activation.items()
            if activation > 0.5
        )
        
        return {
            'matched_heuristic': matched_pattern.pattern_id if matched_pattern else None,
            'recommended_action': matched_pattern.action if matched_pattern else None,
            'differential_changes': differential['changes'],
            'change_magnitude': differential['magnitude'],
            'cross_domain_insights': cross_insights,
            'coherence_score': coherence_check.coherence_score,
            'cognitive_valence': coherence_check.valence.value,
            'dissonances': coherence_check.dissonances,
            'recommendations': coherence_check.recommendations,
            'active_nodes': list(self.mind_state['active_domains']),
            'global_activation': self.multi_node.global_activation
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mind system statistics."""
        return {
            'theory_of_mind': {
                'self_states': sum(len(states) for states in self.theory_of_mind.self_states.values()),
                'tracked_agents': len(self.theory_of_mind.other_states),
                'perspective_switches': len(self.theory_of_mind.perspective_history)
            },
            'heuristics': {
                'total_patterns': len(self.heuristic_analyzer.patterns),
                'avg_success_rate': np.mean([p.success_rate for p in self.heuristic_analyzer.patterns.values()]) if self.heuristic_analyzer.patterns else 0.0,
                'total_usage': sum(p.usage_count for p in self.heuristic_analyzer.patterns.values())
            },
            'multi_node': {
                'total_nodes': len(self.multi_node.nodes),
                'total_connections': len(self.multi_node.connection_strengths),
                'avg_activation': np.mean(list(self.multi_node.global_activation.values())) if self.multi_node.global_activation else 0.0
            },
            'coherence': {
                'current_coherence': self.mind_state['coherence'],
                'dissonance_level': self.mind_state['dissonance_level'],
                'total_beliefs': len(self.coherence_analyzer.beliefs),
                'total_contradictions': sum(len(c) for c in self.coherence_analyzer.contradicts.values())
            }
        }
    
    def print_stats(self):
        """Print mind system statistics."""
        if not self.verbose:
            return
        
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("MIND SYSTEM STATISTICS".center(80))
        print("="*80)
        
        print("\n[ToM] Theory of Mind:")
        print(f"  Self States: {stats['theory_of_mind']['self_states']}")
        print(f"  Tracked Agents: {stats['theory_of_mind']['tracked_agents']}")
        print(f"  Perspective Switches: {stats['theory_of_mind']['perspective_switches']}")
        
        print("\n[HDA] Heuristic Analysis:")
        print(f"  Total Patterns: {stats['heuristics']['total_patterns']}")
        print(f"  Avg Success Rate: {stats['heuristics']['avg_success_rate']:.2%}")
        print(f"  Total Usage: {stats['heuristics']['total_usage']}")
        
        print("\n[MNCP] Multi-Node Network:")
        print(f"  Total Nodes: {stats['multi_node']['total_nodes']}")
        print(f"  Total Connections: {stats['multi_node']['total_connections']}")
        print(f"  Avg Activation: {stats['multi_node']['avg_activation']:.2f}")
        
        print("\n[CCA] Cognitive Coherence:")
        print(f"  Current Coherence: {stats['coherence']['current_coherence']:.2%}")
        print(f"  Dissonance Level: {stats['coherence']['dissonance_level']:.2%}")
        print(f"  Total Beliefs: {stats['coherence']['total_beliefs']}")
        print(f"  Contradictions: {stats['coherence']['total_contradictions']}")
        
        print("\n[STATE] Mind State:")
        print(f"  Active Domains: {len(self.mind_state['active_domains'])}")
        print(f"  Current Perspective: {self.mind_state['perspective']}")
        
        print("="*80 + "\n")
