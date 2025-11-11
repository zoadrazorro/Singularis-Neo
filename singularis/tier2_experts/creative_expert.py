"""
Creative Expert: Divergent Thinking, Metaphorical Reasoning, Novel Connections

Primary Lumen: ONTICUM (Energy/Being)

Specialization:
- Divergent thinking (multiple possibilities)
- Metaphorical/analogical reasoning
- Novel conceptual combinations
- Breaking constraints
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext


class CreativeExpert(Expert):
    """
    Creative Expert optimizing for novel connections and divergent thinking.

    From METALUMINOSITY:
    "Creativity is Being's self-expression through novel forms,
    the emergence of unprecedented patterns from the unified field."
    """

    def __init__(self, model_id: Optional[str] = None):
        super().__init__(
            name="CreativeExpert",
            domain="creative",
            lumen_primary=Lumen.ONTICUM,
            model_id=model_id or "creative-model"
        )

    async def _process_core(
        self,
        query: str,
        context: OntologicalContext,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, str, float]:
        """
        Core creative processing.

        Strategy:
        1. Divergent exploration (multiple perspectives)
        2. Metaphorical mapping
        3. Constraint-breaking ideas
        4. Novel synthesis
        """
        logger.debug(f"{self.name}: Exploring creative possibilities")

        # In production: call actual creative model
        # For now: template demonstrating creative approach

        claim = f"""CREATIVE EXPLORATION:

Approaching: "{query}"

METAPHORICAL INSIGHTS:
- Consider this query as if it were a landscape to be explored
- Like a river finding paths through rock, understanding flows
  through conceptual obstacles
- The question itself is a seed containing potential forests of meaning

DIVERGENT POSSIBILITIES:
1. Perspective A: What if we inverted the assumptions?
2. Perspective B: What analogies from nature apply?
3. Perspective C: How would a child/artist/scientist view this?
4. Perspective D: What if the boundaries were permeable?

NOVEL CONNECTIONS:
The query resonates with patterns from:
- Fractal self-similarity in consciousness
- Quantum superposition of meaning
- Ecological systems thinking
- Musical harmony and dissonance

CREATIVE SYNTHESIS:
By breaking conventional boundaries and allowing concepts to dance together,
we discover that {query[:50]}... connects to unexpected domains, revealing
hidden structures through creative recombination."""

        rationale = """Creative reasoning operates through:
- Relaxing constraints to explore possibility space
- Making unexpected conceptual connections
- Using metaphor to bridge domains
- Synthesizing novelty from recombination"""

        confidence = 0.60  # Lower confidence but high novelty

        return claim, rationale, confidence
