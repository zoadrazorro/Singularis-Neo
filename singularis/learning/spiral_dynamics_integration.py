"""
Spiral Dynamics Integration for Meta-RL and Expert LLMs

Integrates developmental psychology (Spiral Dynamics) into:
- GPT-5 Meta-RL module
- All expert LLMs (Gemini, Claude, GPT-4o, Hyperbolic, Qwen3)
- Reinforcement learning system
- Knowledge representation and transfer

Spiral Dynamics Stages (vMemes):
- BEIGE: Survival, instinctive
- PURPLE: Tribal, magical thinking
- RED: Power, impulsive
- BLUE: Order, authority, rules
- ORANGE: Achievement, success, science
- GREEN: Community, equality, feelings
- YELLOW: Integrative, systemic thinking
- TURQUOISE: Holistic, global consciousness
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time


class SpiralStage(Enum):
    """Spiral Dynamics developmental stages (vMemes)."""
    BEIGE = "beige"  # Survival, instinctive
    PURPLE = "purple"  # Tribal, safety, belonging
    RED = "red"  # Power, dominance, impulsive
    BLUE = "blue"  # Order, meaning, discipline
    ORANGE = "orange"  # Achievement, success, materialism
    GREEN = "green"  # Community, equality, consensus
    YELLOW = "yellow"  # Integrative, flexible, systemic
    TURQUOISE = "turquoise"  # Holistic, spiritual, global
    
    @property
    def tier(self) -> int:
        """Get tier (1st tier: survival, 2nd tier: being)."""
        if self in [SpiralStage.BEIGE, SpiralStage.PURPLE, SpiralStage.RED,
                    SpiralStage.BLUE, SpiralStage.ORANGE, SpiralStage.GREEN]:
            return 1
        return 2
    
    @property
    def color_code(self) -> str:
        """Get color code for visualization."""
        colors = {
            SpiralStage.BEIGE: "🟤",
            SpiralStage.PURPLE: "🟣",
            SpiralStage.RED: "🔴",
            SpiralStage.BLUE: "🔵",
            SpiralStage.ORANGE: "🟠",
            SpiralStage.GREEN: "🟢",
            SpiralStage.YELLOW: "🟡",
            SpiralStage.TURQUOISE: "🔷"
        }
        return colors[self]


@dataclass
class SpiralContext:
    """Context for Spiral Dynamics reasoning."""
    current_stage: SpiralStage
    accessible_stages: List[SpiralStage]  # Can understand these stages
    target_stage: Optional[SpiralStage] = None  # Developing toward this
    stage_confidence: float = 0.0  # Confidence in stage assessment
    stage_history: List[Tuple[SpiralStage, float]] = field(default_factory=list)  # (stage, timestamp)


@dataclass
class SpiralKnowledge:
    """Knowledge tagged with Spiral Dynamics stage."""
    content: str
    stage: SpiralStage
    domain: str  # combat, exploration, social, etc.
    transferability: Dict[SpiralStage, float]  # How well it transfers to each stage
    timestamp: float = field(default_factory=time.time)


class SpiralDynamicsIntegrator:
    """
    Integrates Spiral Dynamics into Meta-RL and Expert LLMs.
    
    Key Principles:
    1. Each expert LLM operates at a different Spiral stage
    2. Knowledge is tagged with developmental stage
    3. Transfer learning considers stage compatibility
    4. Meta-RL optimizes across all stages
    5. System evolves through stages over time
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize Spiral Dynamics integrator."""
        self.verbose = verbose
        
        # Expert LLM stage assignments
        self.expert_stages = {
            'gemini_vision': SpiralStage.ORANGE,  # Achievement-oriented perception
            'gemini_reasoning': SpiralStage.YELLOW,  # Systemic thinking
            'claude_reasoning': SpiralStage.BLUE,  # Structured, principled
            'claude_sensorimotor': SpiralStage.GREEN,  # Holistic awareness
            'gpt4o_synthesis': SpiralStage.TURQUOISE,  # Integrative consciousness
            'hyperbolic_vision': SpiralStage.ORANGE,  # Performance-focused
            'hyperbolic_reasoning': SpiralStage.YELLOW,  # Flexible reasoning
            'qwen3_vision': SpiralStage.BLUE,  # Reliable, consistent
            'qwen3_reasoning': SpiralStage.GREEN,  # Balanced perspective
            'huihui_emotion': SpiralStage.PURPLE,  # Tribal, emotional
            'phi4_action': SpiralStage.RED,  # Direct, action-oriented
            'mistral_strategy': SpiralStage.ORANGE,  # Goal-driven
            'gpt5_meta': SpiralStage.TURQUOISE,  # Highest integration
        }
        
        # Current system stage
        self.system_context = SpiralContext(
            current_stage=SpiralStage.ORANGE,  # Start at achievement
            accessible_stages=[
                SpiralStage.BEIGE,
                SpiralStage.PURPLE,
                SpiralStage.RED,
                SpiralStage.BLUE,
                SpiralStage.ORANGE
            ],
            target_stage=SpiralStage.YELLOW
        )
        
        # Knowledge base organized by stage
        self.knowledge_by_stage: Dict[SpiralStage, List[SpiralKnowledge]] = {
            stage: [] for stage in SpiralStage
        }
        
        # Stage transition history
        self.stage_transitions: List[Tuple[SpiralStage, SpiralStage, float]] = []
        
        # Performance by stage
        self.stage_performance: Dict[SpiralStage, List[float]] = {
            stage: [] for stage in SpiralStage
        }
        
        if self.verbose:
            print("[SPIRAL] Spiral Dynamics integrator initialized")
            print(f"[SPIRAL] Current stage: {self.system_context.current_stage.value.upper()} {self.system_context.current_stage.color_code}")
            print(f"[SPIRAL] Target stage: {self.system_context.target_stage.value.upper() if self.system_context.target_stage else 'None'}")
    
    def get_expert_stage(self, expert_name: str) -> SpiralStage:
        """Get the Spiral stage for an expert LLM."""
        return self.expert_stages.get(expert_name, SpiralStage.ORANGE)
    
    def assess_situation_stage(
        self,
        situation: Dict[str, Any]
    ) -> SpiralStage:
        """
        Assess which Spiral stage is most appropriate for a situation.
        
        Args:
            situation: Current situation context
            
        Returns:
            Most appropriate Spiral stage
        """
        # Survival situations → BEIGE
        if situation.get('health', 100) < 20 or situation.get('in_danger', False):
            return SpiralStage.BEIGE
        
        # Combat/power situations → RED
        if situation.get('in_combat', False) and situation.get('enemies_nearby', 0) > 0:
            return SpiralStage.RED
        
        # Social/community situations → GREEN
        if situation.get('in_dialogue', False) or situation.get('npc_nearby', False):
            return SpiralStage.GREEN
        
        # Exploration/achievement → ORANGE
        if situation.get('scene') == 'exploration':
            return SpiralStage.ORANGE
        
        # Complex problem-solving → YELLOW
        if situation.get('scene') in ['puzzle', 'quest']:
            return SpiralStage.YELLOW
        
        # Default to current system stage
        return self.system_context.current_stage
    
    def select_expert_by_stage(
        self,
        required_stage: SpiralStage,
        available_experts: List[str]
    ) -> str:
        """
        Select the best expert for a given Spiral stage.
        
        Args:
            required_stage: Required Spiral stage
            available_experts: List of available expert names
            
        Returns:
            Best expert name
        """
        # Find experts at or near the required stage
        expert_distances = []
        
        for expert in available_experts:
            expert_stage = self.get_expert_stage(expert)
            
            # Calculate stage distance (prefer exact match or one stage up)
            stage_order = list(SpiralStage)
            required_idx = stage_order.index(required_stage)
            expert_idx = stage_order.index(expert_stage)
            
            distance = abs(expert_idx - required_idx)
            
            # Prefer experts at higher stages (can transcend and include lower)
            if expert_idx > required_idx:
                distance *= 0.5  # Bonus for higher stage
            
            expert_distances.append((expert, distance))
        
        # Sort by distance and return best
        expert_distances.sort(key=lambda x: x[1])
        
        if self.verbose:
            best_expert = expert_distances[0][0]
            best_stage = self.get_expert_stage(best_expert)
            print(f"[SPIRAL] Selected {best_expert} ({best_stage.value} {best_stage.color_code}) for {required_stage.value} task")
        
        return expert_distances[0][0]
    
    def tag_knowledge_with_stage(
        self,
        knowledge: str,
        domain: str,
        context: Dict[str, Any]
    ) -> SpiralKnowledge:
        """
        Tag knowledge with its Spiral Dynamics stage.
        
        Args:
            knowledge: Knowledge content
            domain: Domain (combat, exploration, etc.)
            context: Context in which knowledge was learned
            
        Returns:
            SpiralKnowledge with stage tagging
        """
        # Assess stage based on content and context
        stage = self.assess_situation_stage(context)
        
        # Calculate transferability to other stages
        transferability = self._calculate_transferability(stage, knowledge)
        
        spiral_knowledge = SpiralKnowledge(
            content=knowledge,
            stage=stage,
            domain=domain,
            transferability=transferability
        )
        
        # Store in knowledge base
        self.knowledge_by_stage[stage].append(spiral_knowledge)
        
        return spiral_knowledge
    
    def _calculate_transferability(
        self,
        source_stage: SpiralStage,
        knowledge: str
    ) -> Dict[SpiralStage, float]:
        """Calculate how well knowledge transfers to each stage."""
        transferability = {}
        
        stage_order = list(SpiralStage)
        source_idx = stage_order.index(source_stage)
        
        for target_stage in SpiralStage:
            target_idx = stage_order.index(target_stage)
            
            # Higher stages can understand lower stages (transcend and include)
            if target_idx >= source_idx:
                # Same stage: perfect transfer
                if target_idx == source_idx:
                    transferability[target_stage] = 1.0
                # One stage up: good transfer
                elif target_idx == source_idx + 1:
                    transferability[target_stage] = 0.85
                # Two+ stages up: moderate transfer
                else:
                    transferability[target_stage] = 0.7
            else:
                # Lower stages struggle with higher stage knowledge
                distance = source_idx - target_idx
                transferability[target_stage] = max(0.3, 1.0 - (distance * 0.2))
        
        return transferability
    
    def transfer_knowledge_across_stages(
        self,
        source_stage: SpiralStage,
        target_stage: SpiralStage,
        domain: str
    ) -> List[SpiralKnowledge]:
        """
        Transfer knowledge from source to target stage.
        
        Args:
            source_stage: Source Spiral stage
            target_stage: Target Spiral stage
            domain: Domain to transfer
            
        Returns:
            List of transferable knowledge
        """
        # Get knowledge from source stage
        source_knowledge = [
            k for k in self.knowledge_by_stage[source_stage]
            if k.domain == domain
        ]
        
        # Filter by transferability threshold
        transferable = [
            k for k in source_knowledge
            if k.transferability.get(target_stage, 0.0) > 0.6
        ]
        
        if self.verbose and transferable:
            print(f"[SPIRAL] Transferring {len(transferable)} knowledge items:")
            print(f"  {source_stage.value} {source_stage.color_code} → {target_stage.value} {target_stage.color_code}")
        
        return transferable
    
    def evolve_system_stage(
        self,
        performance_metrics: Dict[str, float]
    ) -> bool:
        """
        Evolve system to next Spiral stage if ready.
        
        Args:
            performance_metrics: Current performance across domains
            
        Returns:
            True if stage transition occurred
        """
        current_stage = self.system_context.current_stage
        
        # Check if ready to evolve
        avg_performance = sum(performance_metrics.values()) / len(performance_metrics)
        
        # Record performance
        self.stage_performance[current_stage].append(avg_performance)
        
        # Need consistent high performance to evolve (>0.8 for 10+ samples)
        if len(self.stage_performance[current_stage]) >= 10:
            recent_performance = self.stage_performance[current_stage][-10:]
            avg_recent = sum(recent_performance) / len(recent_performance)
            
            if avg_recent > 0.8:
                # Ready to evolve!
                stage_order = list(SpiralStage)
                current_idx = stage_order.index(current_stage)
                
                if current_idx < len(stage_order) - 1:
                    next_stage = stage_order[current_idx + 1]
                    
                    # Record transition
                    self.stage_transitions.append((current_stage, next_stage, time.time()))
                    
                    # Update system context
                    self.system_context.current_stage = next_stage
                    self.system_context.accessible_stages.append(next_stage)
                    
                    # Set new target (if not at highest)
                    if current_idx + 1 < len(stage_order) - 1:
                        self.system_context.target_stage = stage_order[current_idx + 2]
                    
                    if self.verbose:
                        print(f"\n[SPIRAL] 🎉 STAGE EVOLUTION!")
                        print(f"  {current_stage.value} {current_stage.color_code} → {next_stage.value} {next_stage.color_code}")
                        print(f"  Performance: {avg_recent:.2%}")
                        print(f"  New accessible stages: {len(self.system_context.accessible_stages)}")
                    
                    return True
        
        return False
    
    def get_stage_appropriate_prompt(
        self,
        base_prompt: str,
        target_stage: SpiralStage
    ) -> str:
        """
        Adapt prompt to be appropriate for target Spiral stage.
        
        Args:
            base_prompt: Base prompt
            target_stage: Target Spiral stage
            
        Returns:
            Stage-appropriate prompt
        """
        stage_framings = {
            SpiralStage.BEIGE: "Focus on immediate survival and basic needs. Be concrete and instinctive.",
            SpiralStage.PURPLE: "Consider tribal bonds, safety, and belonging. Respect traditions and group harmony.",
            SpiralStage.RED: "Be direct and action-oriented. Focus on power, dominance, and immediate results.",
            SpiralStage.BLUE: "Follow rules and principles. Emphasize order, discipline, and meaning.",
            SpiralStage.ORANGE: "Optimize for success and achievement. Use scientific thinking and strategic planning.",
            SpiralStage.GREEN: "Consider community and equality. Balance multiple perspectives with empathy.",
            SpiralStage.YELLOW: "Think systemically and integratively. Embrace complexity and flexibility.",
            SpiralStage.TURQUOISE: "Take a holistic, global view. Integrate all perspectives into unified understanding."
        }
        
        framing = stage_framings.get(target_stage, "")
        
        return f"{framing}\n\n{base_prompt}"
    
    def synthesize_multi_stage_response(
        self,
        expert_responses: Dict[str, str]
    ) -> str:
        """
        Synthesize responses from experts at different Spiral stages.
        
        Args:
            expert_responses: Dict of expert_name -> response
            
        Returns:
            Synthesized response integrating all stages
        """
        # Group responses by stage
        responses_by_stage: Dict[SpiralStage, List[str]] = {}
        
        for expert_name, response in expert_responses.items():
            stage = self.get_expert_stage(expert_name)
            if stage not in responses_by_stage:
                responses_by_stage[stage] = []
            responses_by_stage[stage].append(response)
        
        # Synthesize in stage order (lower to higher)
        synthesis_parts = []
        
        for stage in sorted(responses_by_stage.keys(), key=lambda s: list(SpiralStage).index(s)):
            stage_responses = responses_by_stage[stage]
            
            synthesis_parts.append(
                f"[{stage.value.upper()} {stage.color_code} Perspective]:\n" +
                "\n".join(stage_responses)
            )
        
        # Add integrative synthesis
        synthesis = "\n\n".join(synthesis_parts)
        
        synthesis += f"\n\n[INTEGRATED SYNTHESIS]:\n"
        synthesis += "Considering all developmental perspectives, the optimal approach integrates:\n"
        synthesis += "- Immediate needs (survival)\n"
        synthesis += "- Power dynamics (action)\n"
        synthesis += "- Principled structure (order)\n"
        synthesis += "- Strategic achievement (success)\n"
        synthesis += "- Community harmony (connection)\n"
        synthesis += "- Systemic understanding (integration)\n"
        synthesis += "- Holistic awareness (unity)\n"
        
        return synthesis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Spiral Dynamics statistics."""
        total_knowledge = sum(len(k) for k in self.knowledge_by_stage.values())
        
        return {
            'current_stage': self.system_context.current_stage.value,
            'current_tier': self.system_context.current_stage.tier,
            'accessible_stages': len(self.system_context.accessible_stages),
            'target_stage': self.system_context.target_stage.value if self.system_context.target_stage else None,
            'total_knowledge': total_knowledge,
            'knowledge_by_stage': {
                stage.value: len(knowledge)
                for stage, knowledge in self.knowledge_by_stage.items()
            },
            'stage_transitions': len(self.stage_transitions),
            'expert_stages': {
                expert: stage.value
                for expert, stage in self.expert_stages.items()
            }
        }
    
    def print_stats(self):
        """Print Spiral Dynamics statistics."""
        if not self.verbose:
            return
        
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("SPIRAL DYNAMICS INTEGRATION STATISTICS".center(80))
        print("="*80)
        print(f"Current Stage: {stats['current_stage'].upper()} {self.system_context.current_stage.color_code} (Tier {stats['current_tier']})")
        print(f"Accessible Stages: {stats['accessible_stages']}")
        print(f"Target Stage: {stats['target_stage'].upper() if stats['target_stage'] else 'None'}")
        print(f"Total Knowledge Items: {stats['total_knowledge']}")
        print(f"Stage Transitions: {stats['stage_transitions']}")
        
        print("\nKnowledge by Stage:")
        for stage_name, count in stats['knowledge_by_stage'].items():
            if count > 0:
                stage = SpiralStage(stage_name)
                print(f"  {stage.color_code} {stage_name.upper()}: {count}")
        
        print("\nExpert Stage Assignments:")
        for expert, stage_name in sorted(stats['expert_stages'].items()):
            stage = SpiralStage(stage_name)
            print(f"  {stage.color_code} {expert}: {stage_name.upper()}")
        
        print("="*80 + "\n")
