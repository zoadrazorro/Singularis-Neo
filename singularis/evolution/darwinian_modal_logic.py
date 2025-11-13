"""
Darwinian Modal Logic System

Uses Gemini Flash 2.0 for modal logic reasoning with evolutionary selection.
Implements possible worlds semantics where decision strategies compete for survival.

Modal operators:
- □ (Necessity): Must be true in all possible worlds
- ◇ (Possibility): True in at least one possible world
- ⊃ (Strict implication): Necessarily implies

Darwinian aspect:
- Strategies that lead to better outcomes survive
- Weak strategies are eliminated
- Mutations create new strategy variants
- Fitness measured by coherence increase and reward
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import random
from loguru import logger


class ModalOperator(Enum):
    """Modal logic operators."""
    NECESSITY = "□"  # Must be true in all worlds
    POSSIBILITY = "◇"  # True in some world
    STRICT_IMPLICATION = "⊃"  # Necessarily implies
    CONTINGENCY = "△"  # Could be true or false


@dataclass
class PossibleWorld:
    """
    A possible world in modal logic.
    
    Represents a complete state of affairs that could obtain.
    """
    world_id: str
    state: Dict[str, Any]
    
    # Accessibility relations (which worlds are accessible from this one)
    accessible_worlds: Set[str] = field(default_factory=set)
    
    # Fitness metrics
    coherence: float = 0.0
    reward: float = 0.0
    survival_score: float = 0.0
    
    # Evolution
    generation: int = 0
    parent_world_id: Optional[str] = None
    mutations: List[str] = field(default_factory=list)


@dataclass
class ModalProposition:
    """A proposition in modal logic."""
    content: str
    operator: ModalOperator
    truth_value: Optional[bool] = None
    
    # Which worlds make this true
    true_in_worlds: Set[str] = field(default_factory=set)
    
    def evaluate(self, worlds: Dict[str, PossibleWorld], current_world_id: str) -> bool:
        """Evaluate proposition in current world."""
        current_world = worlds[current_world_id]
        
        if self.operator == ModalOperator.NECESSITY:
            # True in all accessible worlds
            for world_id in current_world.accessible_worlds:
                if world_id not in self.true_in_worlds:
                    return False
            return True
        
        elif self.operator == ModalOperator.POSSIBILITY:
            # True in at least one accessible world
            for world_id in current_world.accessible_worlds:
                if world_id in self.true_in_worlds:
                    return True
            return False
        
        elif self.operator == ModalOperator.CONTINGENCY:
            # True in some but not all accessible worlds
            true_count = sum(1 for w in current_world.accessible_worlds if w in self.true_in_worlds)
            return 0 < true_count < len(current_world.accessible_worlds)
        
        return False


class DarwinianModalLogic:
    """
    Darwinian modal logic system using Gemini Flash 2.0.
    
    Evolves decision strategies through modal reasoning and natural selection.
    """
    
    def __init__(self, gemini_client):
        """
        Initialize Darwinian modal logic.
        
        Args:
            gemini_client: Gemini Flash 2.0 client
        """
        self.gemini = gemini_client
        
        # Possible worlds
        self.worlds: Dict[str, PossibleWorld] = {}
        self.current_world_id: Optional[str] = None
        
        # Propositions
        self.propositions: List[ModalProposition] = []
        
        # Evolution parameters
        self.population_size = 10
        self.mutation_rate = 0.15
        self.selection_pressure = 0.7  # Top 70% survive
        
        # Statistics
        self.generation = 0
        self.total_worlds_created = 0
        self.total_worlds_eliminated = 0
        
        logger.info("[DARWINIAN-LOGIC] System initialized")
    
    async def initialize_worlds(self, initial_state: Dict[str, Any]):
        """Initialize population of possible worlds."""
        # Create initial world
        initial_world = PossibleWorld(
            world_id="world_0",
            state=initial_state.copy(),
            coherence=0.5,
            reward=0.0,
            survival_score=0.5,
            generation=0
        )
        
        self.worlds[initial_world.world_id] = initial_world
        self.current_world_id = initial_world.world_id
        self.total_worlds_created = 1
        
        # Generate variant worlds through modal reasoning
        variants = await self._generate_world_variants(initial_world, count=self.population_size - 1)
        
        for variant in variants:
            self.worlds[variant.world_id] = variant
            self.total_worlds_created += 1
            
            # Make worlds accessible to each other
            initial_world.accessible_worlds.add(variant.world_id)
            variant.accessible_worlds.add(initial_world.world_id)
        
        logger.info(f"[DARWINIAN-LOGIC] Initialized {len(self.worlds)} possible worlds")
    
    async def _generate_world_variants(
        self,
        base_world: PossibleWorld,
        count: int
    ) -> List[PossibleWorld]:
        """Generate variant worlds using Gemini Flash 2.0."""
        prompt = f"""Generate {count} variant possible worlds based on this base world:

Base World State:
{self._format_state(base_world.state)}

For each variant, propose a DIFFERENT decision strategy or state configuration.
Each variant should explore a different possibility space.

Format each variant as:
VARIANT N:
Strategy: [decision strategy]
State Changes: [key differences from base]
Rationale: [why this variant might succeed]

Generate {count} distinct variants."""
        
        response = await self.gemini.generate(
            prompt=prompt,
            temperature=0.9,  # High for diversity
            max_tokens=2048
        )
        
        # Parse variants
        variants = self._parse_variants(response, base_world)
        
        return variants
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state for display."""
        lines = []
        for key, value in state.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def _parse_variants(
        self,
        response: str,
        base_world: PossibleWorld
    ) -> List[PossibleWorld]:
        """Parse variant worlds from Gemini response."""
        variants = []
        lines = response.split('\n')
        
        current_variant = None
        variant_count = 0
        
        for line in lines:
            if line.startswith('VARIANT'):
                if current_variant:
                    variants.append(current_variant)
                
                variant_count += 1
                current_variant = PossibleWorld(
                    world_id=f"world_{self.total_worlds_created + variant_count}",
                    state=base_world.state.copy(),
                    coherence=0.5 + random.uniform(-0.1, 0.1),
                    reward=0.0,
                    survival_score=0.5,
                    generation=base_world.generation + 1,
                    parent_world_id=base_world.world_id
                )
            
            elif line.startswith('Strategy:') and current_variant:
                strategy = line.replace('Strategy:', '').strip()
                current_variant.state['strategy'] = strategy
            
            elif line.startswith('State Changes:') and current_variant:
                changes = line.replace('State Changes:', '').strip()
                current_variant.mutations.append(changes)
        
        if current_variant:
            variants.append(current_variant)
        
        return variants
    
    async def evaluate_worlds(
        self,
        context: Dict[str, Any],
        outcomes: Dict[str, Any]
    ):
        """
        Evaluate fitness of all worlds based on outcomes.
        
        Args:
            context: Current context
            outcomes: Observed outcomes
        """
        # Evaluate each world
        for world in self.worlds.values():
            # Update coherence and reward based on how well world's strategy performed
            if 'coherence_delta' in outcomes:
                world.coherence += outcomes['coherence_delta']
            
            if 'reward' in outcomes:
                world.reward += outcomes['reward']
            
            # Compute survival score (fitness)
            world.survival_score = (
                world.coherence * 0.5 +
                world.reward * 0.3 +
                (1.0 / (world.generation + 1)) * 0.2  # Slight bias for newer generations
            )
        
        logger.info(f"[DARWINIAN-LOGIC] Evaluated {len(self.worlds)} worlds")
    
    async def natural_selection(self):
        """
        Perform natural selection on worlds.
        
        - Eliminate low-fitness worlds
        - Reproduce high-fitness worlds with mutations
        - Maintain population size
        """
        self.generation += 1
        
        # Sort by survival score
        sorted_worlds = sorted(
            self.worlds.values(),
            key=lambda w: w.survival_score,
            reverse=True
        )
        
        # Select survivors (top X%)
        survival_cutoff = int(len(sorted_worlds) * self.selection_pressure)
        survivors = sorted_worlds[:survival_cutoff]
        eliminated = sorted_worlds[survival_cutoff:]
        
        # Remove eliminated worlds
        for world in eliminated:
            del self.worlds[world.world_id]
            self.total_worlds_eliminated += 1
        
        logger.info(f"[DARWINIAN-LOGIC] Generation {self.generation}: {len(survivors)} survivors, {len(eliminated)} eliminated")
        
        # Reproduce survivors to maintain population
        offspring_needed = self.population_size - len(survivors)
        
        if offspring_needed > 0:
            # Select parents weighted by fitness
            parents = random.choices(
                survivors,
                weights=[w.survival_score for w in survivors],
                k=offspring_needed
            )
            
            # Generate offspring with mutations
            for parent in parents:
                offspring = await self._mutate_world(parent)
                self.worlds[offspring.world_id] = offspring
                self.total_worlds_created += 1
                
                # Update accessibility
                for world in self.worlds.values():
                    if world.world_id != offspring.world_id:
                        world.accessible_worlds.add(offspring.world_id)
                        offspring.accessible_worlds.add(world.world_id)
        
        logger.info(f"[DARWINIAN-LOGIC] Generated {offspring_needed} offspring")
    
    async def _mutate_world(self, parent: PossibleWorld) -> PossibleWorld:
        """Create mutated offspring from parent world."""
        # Decide what to mutate
        if random.random() < self.mutation_rate:
            # Mutate strategy
            prompt = f"""Mutate this decision strategy to create a variant:

Parent Strategy: {parent.state.get('strategy', 'explore and adapt')}
Parent Fitness: {parent.survival_score:.3f}

Create a MUTATED version that:
1. Keeps the core idea but varies the approach
2. Might improve on weaknesses
3. Explores a slightly different possibility

Mutated Strategy:"""
            
            response = await self.gemini.generate(
                prompt=prompt,
                temperature=0.85,
                max_tokens=256
            )
            
            mutated_strategy = response.strip()
        else:
            mutated_strategy = parent.state.get('strategy', 'explore')
        
        # Create offspring
        offspring = PossibleWorld(
            world_id=f"world_{self.total_worlds_created + 1}",
            state=parent.state.copy(),
            coherence=parent.coherence + random.uniform(-0.05, 0.05),
            reward=0.0,  # Reset reward for new generation
            survival_score=parent.survival_score * 0.9,  # Inherit some fitness
            generation=self.generation,
            parent_world_id=parent.world_id,
            mutations=[f"Mutated from {parent.world_id}"]
        )
        
        offspring.state['strategy'] = mutated_strategy
        
        return offspring
    
    async def modal_reasoning(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform modal reasoning across possible worlds.
        
        Args:
            query: Question to reason about
            context: Current context
        
        Returns:
            Modal reasoning results
        """
        # Build modal reasoning prompt
        prompt = self._build_modal_prompt(query, context)
        
        # Get Gemini's modal analysis
        response = await self.gemini.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1536
        )
        
        # Parse modal propositions
        propositions = self._parse_modal_propositions(response)
        
        # Evaluate propositions across worlds
        results = {
            'query': query,
            'propositions': [],
            'necessity_claims': [],
            'possibility_claims': [],
            'best_world': None
        }
        
        for prop in propositions:
            evaluation = {
                'content': prop.content,
                'operator': prop.operator.value,
                'truth_value': prop.evaluate(self.worlds, self.current_world_id)
            }
            results['propositions'].append(evaluation)
            
            if prop.operator == ModalOperator.NECESSITY and evaluation['truth_value']:
                results['necessity_claims'].append(prop.content)
            elif prop.operator == ModalOperator.POSSIBILITY and evaluation['truth_value']:
                results['possibility_claims'].append(prop.content)
        
        # Find best world
        best_world = max(self.worlds.values(), key=lambda w: w.survival_score)
        results['best_world'] = {
            'world_id': best_world.world_id,
            'strategy': best_world.state.get('strategy', 'unknown'),
            'survival_score': best_world.survival_score,
            'generation': best_world.generation
        }
        
        return results
    
    def _build_modal_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build modal reasoning prompt."""
        # Get top 3 worlds
        top_worlds = sorted(
            self.worlds.values(),
            key=lambda w: w.survival_score,
            reverse=True
        )[:3]
        
        prompt_parts = [
            "MODAL LOGIC REASONING",
            "=" * 70,
            "",
            f"Query: {query}",
            "",
            "Context:",
            self._format_state(context),
            "",
            "Top Possible Worlds:",
        ]
        
        for i, world in enumerate(top_worlds, 1):
            prompt_parts.append(f"\nWorld {i} (fitness: {world.survival_score:.3f}):")
            prompt_parts.append(f"  Strategy: {world.state.get('strategy', 'unknown')}")
            prompt_parts.append(f"  Generation: {world.generation}")
        
        prompt_parts.extend([
            "",
            "Provide modal logic analysis using:",
            "□ (NECESSITY): What MUST be true across all viable worlds?",
            "◇ (POSSIBILITY): What COULD be true in some world?",
            "△ (CONTINGENCY): What might or might not be true?",
            "",
            "Format:",
            "□ [Necessary claim]",
            "◇ [Possible claim]",
            "△ [Contingent claim]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_modal_propositions(self, response: str) -> List[ModalProposition]:
        """Parse modal propositions from response."""
        propositions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('□'):
                content = line[1:].strip()
                propositions.append(ModalProposition(
                    content=content,
                    operator=ModalOperator.NECESSITY
                ))
            
            elif line.startswith('◇'):
                content = line[1:].strip()
                propositions.append(ModalProposition(
                    content=content,
                    operator=ModalOperator.POSSIBILITY
                ))
            
            elif line.startswith('△'):
                content = line[1:].strip()
                propositions.append(ModalProposition(
                    content=content,
                    operator=ModalOperator.CONTINGENCY
                ))
        
        return propositions
    
    def get_best_strategy(self) -> str:
        """Get strategy from highest-fitness world."""
        if not self.worlds:
            return "explore"
        
        best_world = max(self.worlds.values(), key=lambda w: w.survival_score)
        return best_world.state.get('strategy', 'explore')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.worlds:
            return {'error': 'No worlds initialized'}
        
        avg_fitness = sum(w.survival_score for w in self.worlds.values()) / len(self.worlds)
        best_world = max(self.worlds.values(), key=lambda w: w.survival_score)
        
        return {
            'generation': self.generation,
            'active_worlds': len(self.worlds),
            'total_created': self.total_worlds_created,
            'total_eliminated': self.total_worlds_eliminated,
            'average_fitness': float(avg_fitness),
            'best_fitness': float(best_world.survival_score),
            'best_strategy': best_world.state.get('strategy', 'unknown'),
            'best_generation': best_world.generation
        }
