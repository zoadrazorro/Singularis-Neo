"""
Analytic Evolution Heuristic

Uses Claude Haiku for fast analytic reasoning about evolutionary trajectories.
Complements Darwinian modal logic with analytical decomposition and synthesis.

Key features:
- Rapid analysis of decision patterns
- Evolutionary trajectory prediction
- Heuristic decomposition (break complex into simple)
- Synthesis (combine simple into complex)
- Fitness landscape navigation
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger


@dataclass
class AnalyticNode:
    """
    A node in the analytic evolution tree.
    
    Represents a decomposed component of a complex decision.
    """
    node_id: str
    content: str
    
    # Decomposition
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    
    # Analysis
    complexity: float = 0.5  # 0.0 = simple, 1.0 = complex
    clarity: float = 0.5  # How well understood
    utility: float = 0.5  # How useful for decisions
    
    # Evolution
    generation: int = 0
    fitness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'content': self.content[:100],
            'depth': self.depth,
            'complexity': float(self.complexity),
            'clarity': float(self.clarity),
            'utility': float(self.utility),
            'fitness': float(self.fitness)
        }


@dataclass
class EvolutionaryTrajectory:
    """
    A predicted evolutionary trajectory.
    
    Represents a path through the fitness landscape.
    """
    trajectory_id: str
    steps: List[str]  # Sequence of decisions/states
    
    # Prediction
    predicted_fitness: float = 0.0
    confidence: float = 0.0
    
    # Analysis
    bottlenecks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


class AnalyticEvolution:
    """
    Analytic evolution heuristic using Claude Haiku.
    
    Provides fast analytical reasoning to guide evolutionary search.
    """
    
    def __init__(self, claude_haiku_client):
        """
        Initialize analytic evolution.
        
        Args:
            claude_haiku_client: Claude Haiku client
        """
        self.claude = claude_haiku_client
        
        # Analytic tree
        self.nodes: Dict[str, AnalyticNode] = {}
        self.root_node_id: Optional[str] = None
        
        # Trajectories
        self.trajectories: List[EvolutionaryTrajectory] = []
        
        # Statistics
        self.total_analyses = 0
        self.total_decompositions = 0
        self.total_syntheses = 0
        
        logger.info("[ANALYTIC-EVOLUTION] System initialized")
    
    async def analyze_decision(
        self,
        decision: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analytically decompose a decision into components.
        
        Args:
            decision: Decision to analyze
            context: Current context
        
        Returns:
            Analysis results
        """
        self.total_analyses += 1
        
        # Build analysis prompt
        prompt = f"""Analytically decompose this decision:

Decision: {decision}

Context:
{self._format_context(context)}

Provide ANALYTICAL DECOMPOSITION:
1. Break the decision into 3-5 fundamental components
2. For each component, assess:
   - Complexity (simple/moderate/complex)
   - Clarity (how well understood)
   - Utility (how useful for the decision)

Format:
COMPONENT 1: [name]
  Description: [what it is]
  Complexity: [simple/moderate/complex]
  Clarity: [low/medium/high]
  Utility: [low/medium/high]

Provide 3-5 components."""
        
        response = await self.claude.generate(
            prompt=prompt,
            temperature=0.5,  # Lower for analytical precision
            max_tokens=1024
        )
        
        # Parse components
        components = self._parse_components(response)
        
        # Create analytic nodes
        if not self.root_node_id:
            root = AnalyticNode(
                node_id="node_0",
                content=decision,
                depth=0,
                generation=0
            )
            self.nodes[root.node_id] = root
            self.root_node_id = root.node_id
        
        # Add component nodes
        for component in components:
            node = AnalyticNode(
                node_id=f"node_{len(self.nodes)}",
                content=component['description'],
                parent_id=self.root_node_id,
                depth=1,
                complexity=component['complexity'],
                clarity=component['clarity'],
                utility=component['utility'],
                generation=0
            )
            
            self.nodes[node.node_id] = node
            self.nodes[self.root_node_id].children_ids.append(node.node_id)
        
        self.total_decompositions += 1
        
        return {
            'decision': decision,
            'components': [c.to_dict() for c in components],
            'total_nodes': len(self.nodes)
        }
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for display."""
        lines = []
        for key, value in context.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def _parse_components(self, response: str) -> List[AnalyticNode]:
        """Parse components from Claude response."""
        components = []
        lines = response.split('\n')
        
        current_component = None
        
        for line in lines:
            if line.startswith('COMPONENT'):
                if current_component:
                    components.append(current_component)
                
                name = line.split(':', 1)[1].strip() if ':' in line else "Component"
                current_component = AnalyticNode(
                    node_id=f"temp_{len(components)}",
                    content=name,
                    depth=1
                )
            
            elif current_component:
                if 'Description:' in line:
                    desc = line.split('Description:', 1)[1].strip()
                    current_component.content = desc
                
                elif 'Complexity:' in line:
                    complexity_str = line.split('Complexity:', 1)[1].strip().lower()
                    if 'simple' in complexity_str:
                        current_component.complexity = 0.3
                    elif 'complex' in complexity_str:
                        current_component.complexity = 0.9
                    else:
                        current_component.complexity = 0.6
                
                elif 'Clarity:' in line:
                    clarity_str = line.split('Clarity:', 1)[1].strip().lower()
                    if 'low' in clarity_str:
                        current_component.clarity = 0.3
                    elif 'high' in clarity_str:
                        current_component.clarity = 0.9
                    else:
                        current_component.clarity = 0.6
                
                elif 'Utility:' in line:
                    utility_str = line.split('Utility:', 1)[1].strip().lower()
                    if 'low' in utility_str:
                        current_component.utility = 0.3
                    elif 'high' in utility_str:
                        current_component.utility = 0.9
                    else:
                        current_component.utility = 0.6
        
        if current_component:
            components.append(current_component)
        
        return components
    
    async def synthesize_strategy(
        self,
        components: List[AnalyticNode]
    ) -> str:
        """
        Synthesize components into coherent strategy.
        
        Args:
            components: Analytic components
        
        Returns:
            Synthesized strategy
        """
        self.total_syntheses += 1
        
        # Build synthesis prompt
        component_descriptions = []
        for comp in components:
            component_descriptions.append(
                f"- {comp.content} (utility: {comp.utility:.2f}, clarity: {comp.clarity:.2f})"
            )
        
        prompt = f"""Synthesize these analytical components into a coherent strategy:

Components:
{chr(10).join(component_descriptions)}

Provide a SYNTHESIZED STRATEGY that:
1. Integrates all high-utility components
2. Addresses low-clarity components
3. Creates a coherent, executable plan

Synthesized Strategy:"""
        
        response = await self.claude.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=512
        )
        
        return response.strip()
    
    async def predict_trajectory(
        self,
        current_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        steps: int = 5
    ) -> EvolutionaryTrajectory:
        """
        Predict evolutionary trajectory from current to goal state.
        
        Args:
            current_state: Current state
            goal_state: Desired goal state
            steps: Number of steps to predict
        
        Returns:
            Predicted trajectory
        """
        prompt = f"""Predict evolutionary trajectory:

Current State:
{self._format_context(current_state)}

Goal State:
{self._format_context(goal_state)}

Predict {steps} intermediate steps to evolve from current to goal.
For each step, identify:
- Action to take
- Expected outcome
- Potential bottleneck
- Opportunity to exploit

Format:
STEP 1:
  Action: [what to do]
  Outcome: [expected result]
  Bottleneck: [potential problem]
  Opportunity: [what to leverage]

Provide {steps} steps."""
        
        response = await self.claude.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1536
        )
        
        # Parse trajectory
        trajectory = self._parse_trajectory(response)
        
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def _parse_trajectory(self, response: str) -> EvolutionaryTrajectory:
        """Parse trajectory from Claude response."""
        trajectory = EvolutionaryTrajectory(
            trajectory_id=f"traj_{len(self.trajectories)}",
            steps=[],
            predicted_fitness=0.5,
            confidence=0.7
        )
        
        lines = response.split('\n')
        current_step = None
        
        for line in lines:
            if line.startswith('STEP'):
                if current_step:
                    trajectory.steps.append(current_step)
                current_step = ""
            
            elif current_step is not None:
                if 'Action:' in line:
                    action = line.split('Action:', 1)[1].strip()
                    current_step = action
                
                elif 'Bottleneck:' in line:
                    bottleneck = line.split('Bottleneck:', 1)[1].strip()
                    trajectory.bottlenecks.append(bottleneck)
                
                elif 'Opportunity:' in line:
                    opportunity = line.split('Opportunity:', 1)[1].strip()
                    trajectory.opportunities.append(opportunity)
        
        if current_step:
            trajectory.steps.append(current_step)
        
        return trajectory
    
    async def evaluate_fitness_landscape(
        self,
        current_position: Dict[str, Any],
        nearby_positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate local fitness landscape.
        
        Args:
            current_position: Current position in state space
            nearby_positions: Nearby positions to evaluate
        
        Returns:
            Landscape analysis
        """
        prompt = f"""Analyze fitness landscape:

Current Position:
{self._format_context(current_position)}

Nearby Positions:
{chr(10).join([f"{i+1}. {self._format_context(pos)}" for i, pos in enumerate(nearby_positions)])}

Analyze:
1. Which direction shows highest fitness gradient?
2. Are there local maxima to avoid?
3. What's the best next move?

Provide concise analysis."""
        
        response = await self.claude.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=768
        )
        
        return {
            'analysis': response,
            'current_position': current_position,
            'nearby_count': len(nearby_positions)
        }
    
    def compute_node_fitness(self, node: AnalyticNode) -> float:
        """Compute fitness of an analytic node."""
        # Fitness = utility * clarity / complexity
        # High utility, high clarity, low complexity = high fitness
        fitness = (node.utility * node.clarity) / (node.complexity + 0.1)
        node.fitness = fitness
        return fitness
    
    def get_high_fitness_nodes(self, limit: int = 5) -> List[AnalyticNode]:
        """Get highest fitness nodes."""
        # Compute fitness for all nodes
        for node in self.nodes.values():
            self.compute_node_fitness(node)
        
        # Sort by fitness
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.fitness,
            reverse=True
        )
        
        return sorted_nodes[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.nodes:
            avg_fitness = 0.0
        else:
            avg_fitness = sum(self.compute_node_fitness(n) for n in self.nodes.values()) / len(self.nodes)
        
        return {
            'total_analyses': self.total_analyses,
            'total_decompositions': self.total_decompositions,
            'total_syntheses': self.total_syntheses,
            'total_nodes': len(self.nodes),
            'total_trajectories': len(self.trajectories),
            'average_node_fitness': float(avg_fitness)
        }
