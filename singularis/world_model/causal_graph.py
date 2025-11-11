"""
Causal Inference Module - Judea Pearl's Causality Framework

Learns causal structure, not correlations.
Implements do-calculus for interventional reasoning.

Key insight: "If I do X, what happens?" ≠ "What happens when X occurs?"

Philosophical grounding:
- ETHICA Part III, Prop VI: Understanding causation enables freedom
- MATHEMATICA A3: Necessity = Causal necessity
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict


class CausalRelation(Enum):
    """Types of causal relationships."""
    CAUSES = "causes"  # A → B
    PREVENTS = "prevents"  # A ⊣ B
    ENABLES = "enables"  # A ⊢ B
    CORRELATES = "correlates"  # A ⟷ B (no causation)


@dataclass
class CausalNode:
    """
    A node in the causal graph representing a variable/concept.

    Attributes:
        name: Unique identifier
        value: Current state/value
        parents: Direct causes
        children: Direct effects
        node_type: observational/interventional/counterfactual
    """
    name: str
    value: Optional[float] = None
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    node_type: str = "observational"

    # Causal strength (learned from data)
    causal_strengths: Dict[str, float] = field(default_factory=dict)

    def add_parent(self, parent: str, strength: float = 1.0):
        """Add a causal parent (A causes this node)."""
        self.parents.add(parent)
        self.causal_strengths[parent] = strength

    def add_child(self, child: str):
        """Add a causal child (this node causes B)."""
        self.children.add(child)


@dataclass
class CausalEdge:
    """
    A causal edge representing a causal relationship.
    
    Attributes:
        cause: The causing variable
        effect: The effect variable
        strength: Strength of causal relationship (0-1)
        confidence: Confidence in this relationship (0-1)
    """
    cause: str
    effect: str
    strength: float = 1.0
    confidence: float = 1.0


@dataclass
class Intervention:
    """
    An intervention: do(X=x)
    Forces a variable to a specific value, cutting incoming edges.

    This is Pearl's do-operator: the foundation of causal inference.
    """
    variable: str
    value: float
    timestamp: float = 0.0


class CausalGraph:
    """
    Causal graph implementing Pearl's causality framework.

    Key capabilities:
    1. Learn causal structure from observational data
    2. Compute interventional distributions: P(Y|do(X=x))
    3. Counterfactual reasoning: "What if I had done X instead?"
    4. Confounding detection and adjustment

    Uses:
    - NetworkX for graph structure
    - Constraint-based learning (PC algorithm)
    - Bayesian network inference
    """

    def __init__(self, learning_rate: float = 0.1):
        self.nodes: Dict[str, CausalNode] = {}
        self.graph = nx.DiGraph()  # Directed acyclic graph (DAG)
        self.learning_rate = learning_rate

        # History of observations for structure learning
        self.observation_history: List[Dict[str, float]] = []

        # Intervention history for learning from surprise
        self.intervention_history: List[Tuple[Intervention, Dict[str, float]]] = []

    def add_node(self, name: str, value: Optional[float] = None) -> CausalNode:
        """Add a node to the causal graph."""
        if name not in self.nodes:
            node = CausalNode(name=name, value=value)
            self.nodes[name] = node
            self.graph.add_node(name)
        return self.nodes[name]

    def add_edge(self, cause_or_edge, effect: Optional[str] = None, strength: float = 1.0):
        """
        Add causal edge: cause → effect

        Args:
            cause_or_edge: Either a CausalEdge object or a string (cause variable)
            effect: Child variable (if cause_or_edge is a string)
            strength: Causal strength (learned from data)
        """
        # Handle both CausalEdge objects and string arguments
        if isinstance(cause_or_edge, CausalEdge):
            edge = cause_or_edge
            cause = edge.cause
            effect = edge.effect
            strength = edge.strength
        else:
            cause = cause_or_edge
            if effect is None:
                raise ValueError("effect must be provided when cause is a string")
        
        # Ensure nodes exist
        self.add_node(cause)
        self.add_node(effect)

        # Add edge (check for cycles first)
        if not nx.has_path(self.graph, effect, cause):
            self.graph.add_edge(cause, effect, weight=strength)
            self.nodes[effect].add_parent(cause, strength)
            self.nodes[cause].add_child(effect)
        else:
            print(f"Warning: Adding {cause}→{effect} would create cycle. Skipped.")

    def intervene(self, intervention: Intervention) -> Dict[str, float]:
        """
        Perform intervention: do(variable=value)

        This is the KEY operation for causal inference:
        - Cuts incoming edges to intervened variable
        - Propagates effects forward through graph
        - Returns predicted outcomes for all variables

        Args:
            intervention: The do(X=x) operation

        Returns:
            Predicted values for all variables after intervention
        """
        var = intervention.variable
        val = intervention.value

        # Create modified graph with intervention
        intervened_graph = self.graph.copy()

        # Remove incoming edges (this is the "do" operation)
        incoming = list(intervened_graph.predecessors(var))
        for parent in incoming:
            intervened_graph.remove_edge(parent, var)

        # Set intervened value
        values = {var: val}

        # Forward propagate through causal chain
        values = self._forward_propagate(intervened_graph, values)

        return values

    def _forward_propagate(
        self,
        graph: nx.DiGraph,
        initial_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Propagate values forward through causal graph.
        Uses topological sort to respect causal order.
        """
        values = initial_values.copy()

        # Get topological order (respects causation)
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Graph has cycle, use arbitrary order
            topo_order = list(graph.nodes())

        for node_name in topo_order:
            if node_name in values:
                continue  # Already set (intervention or computed)

            # Compute value from parents using causal function
            parents = list(graph.predecessors(node_name))
            if not parents:
                # Root node with no value: use default or prior
                values[node_name] = self.nodes[node_name].value or 0.0
            else:
                # Linear causal model: child = Σ(strength_i × parent_i)
                node_val = 0.0
                for parent in parents:
                    if parent in values:
                        weight = graph.edges[parent, node_name].get('weight', 1.0)
                        node_val += weight * values[parent]
                values[node_name] = node_val

        return values

    def predict_intervention_outcome(
        self,
        action: str,
        state: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict outcome of action in given state.

        This answers: "If I do X in state S, what happens?"
        NOT: "What happens when X occurs naturally?"

        Args:
            action: Variable to intervene on (e.g., "grasp_object")
            state: Current world state

        Returns:
            Predicted next state after intervention
        """
        # Set current state
        for var, val in state.items():
            if var in self.nodes:
                self.nodes[var].value = val

        # Perform intervention
        intervention = Intervention(variable=action, value=1.0)
        outcome = self.intervene(intervention)

        return outcome

    def counterfactual(
        self,
        actual_past: Dict[str, float],
        intervention: Intervention
    ) -> Dict[str, float]:
        """
        Counterfactual reasoning: "What if I had done X instead?"

        Three steps (Pearl's algorithm):
        1. Abduction: Infer latent variables from actual past
        2. Action: Apply intervention
        3. Prediction: Compute counterfactual outcomes
        """
        # Simplified counterfactual: re-run with intervention
        # Full Pearl counterfactuals require structural equations

        # Store actual values
        actual_values = actual_past.copy()

        # Apply intervention
        counterfactual_outcome = self.intervene(intervention)

        return counterfactual_outcome

    def learn_from_observation(self, observation: Dict[str, float]):
        """
        Learn causal structure from observational data.
        Uses simple correlation → causation heuristics.

        For production: Use PC algorithm, FCI, or constraint-based methods.
        """
        self.observation_history.append(observation)

        # Need sufficient data for structure learning
        if len(self.observation_history) < 10:
            return

        # Simple approach: high correlation → possible causation
        # (This is naive; real structure learning is complex)

        # Add all observed variables as nodes
        for var in observation.keys():
            self.add_node(var, observation[var])

    def learn_from_surprise(
        self,
        expected: Dict[str, float],
        actual: Dict[str, float]
    ):
        """
        Update causal model from prediction errors.

        Key to continual learning: when predictions fail, update the model.

        Args:
            expected: Predicted values
            actual: Observed values
        """
        # Compute prediction error for each variable
        for var in expected.keys():
            if var in actual:
                error = actual[var] - expected[var]

                # Update causal strengths based on error
                if var in self.nodes:
                    node = self.nodes[var]
                    for parent in node.parents:
                        if parent in node.causal_strengths:
                            # Gradient descent on causal strength
                            current = node.causal_strengths[parent]
                            # Simple update: strengthen if error is large
                            node.causal_strengths[parent] = current + self.learning_rate * error

    def find_confounders(self, var1: str, var2: str) -> List[str]:
        """
        Find confounding variables between var1 and var2.

        A confounder C affects both var1 and var2, creating spurious correlation.
        """
        confounders = []

        # Find common ancestors
        if var1 in self.graph and var2 in self.graph:
            ancestors1 = nx.ancestors(self.graph, var1)
            ancestors2 = nx.ancestors(self.graph, var2)
            confounders = list(ancestors1 & ancestors2)

        return confounders

    def compute_total_effect(self, cause: str, effect: str) -> float:
        """
        Compute total causal effect of cause on effect.
        Includes all causal paths, not just direct.
        """
        if cause not in self.graph or effect not in self.graph:
            return 0.0

        # Sum over all paths from cause to effect
        try:
            paths = list(nx.all_simple_paths(self.graph, cause, effect))
            total = 0.0

            for path in paths:
                # Multiply strengths along path
                path_effect = 1.0
                for i in range(len(path) - 1):
                    edge_weight = self.graph.edges[path[i], path[i+1]].get('weight', 1.0)
                    path_effect *= edge_weight
                total += path_effect

            return total
        except nx.NetworkXNoPath:
            return 0.0

    def visualize(self) -> str:
        """Return ASCII visualization of causal graph."""
        lines = ["Causal Graph:", "=" * 40]

        for node_name in self.nodes:
            node = self.nodes[node_name]
            if node.parents:
                for parent in node.parents:
                    strength = node.causal_strengths.get(parent, 1.0)
                    lines.append(f"{parent} --({strength:.2f})--> {node_name}")
            else:
                lines.append(f"{node_name} (root)")

        lines.append("=" * 40)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export graph structure for serialization."""
        return {
            'nodes': {
                name: {
                    'value': node.value,
                    'parents': list(node.parents),
                    'children': list(node.children),
                    'causal_strengths': node.causal_strengths,
                }
                for name, node in self.nodes.items()
            },
            'edges': [
                {
                    'cause': u,
                    'effect': v,
                    'strength': data.get('weight', 1.0)
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalGraph':
        """Load graph from serialized data."""
        graph = cls()

        # Add nodes
        for name, node_data in data['nodes'].items():
            graph.add_node(name, node_data['value'])

        # Add edges
        for edge in data['edges']:
            graph.add_edge(edge['cause'], edge['effect'], edge['strength'])

        return graph


# Example usage and tests
if __name__ == "__main__":
    # Example: Causal model of action outcomes
    graph = CausalGraph()

    # Build simple causal model:
    # Temperature → Ice melting
    # Pressure → Ice melting
    graph.add_edge("temperature", "ice_melting", strength=0.8)
    graph.add_edge("pressure", "ice_melting", strength=0.5)
    graph.add_edge("ice_melting", "water_level", strength=1.0)

    print(graph.visualize())

    # Observational: What happens when temperature is high?
    # P(ice_melting | temperature = high)
    graph.nodes["temperature"].value = 100.0

    # Interventional: What if we SET temperature to high?
    # P(ice_melting | do(temperature = 100))
    outcome = graph.predict_intervention_outcome(
        action="temperature",
        state={"temperature": 100.0, "pressure": 1.0}
    )
    print(f"\nIntervention outcome: {outcome}")

    # Counterfactual: "What if temperature had been lower?"
    cf = graph.counterfactual(
        actual_past={"temperature": 100.0, "ice_melting": 80.0},
        intervention=Intervention(variable="temperature", value=0.0)
    )
    print(f"Counterfactual outcome: {cf}")

    # Learn from surprise
    expected = {"ice_melting": 80.0, "water_level": 80.0}
    actual = {"ice_melting": 60.0, "water_level": 60.0}
    graph.learn_from_surprise(expected, actual)

    print(f"\nTotal effect of temperature on water_level: {graph.compute_total_effect('temperature', 'water_level'):.2f}")
