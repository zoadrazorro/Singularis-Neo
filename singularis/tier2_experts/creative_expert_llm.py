"""
Creative Expert with LLM Integration

Primary Lumen: ONTICUM (Energy/Power/Existence)

Specialization:
- Novel idea generation
- Divergent thinking
- Conceptual blending
- Innovative solutions
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface


class CreativeExpertLLM(Expert):
    """
    Creative Expert with LLM integration.

    From ETHICA UNIVERSALIS:
    "The power of the mind is defined by the power of understanding alone."
    Creative power emerges from the mind's capacity to form new adequate ideas.
    """

    def __init__(
        self,
        llm_interface: ExpertLLMInterface,
        model_id: Optional[str] = None
    ):
        """
        Initialize creative expert with LLM.
        
        Args:
            llm_interface: LLM interface for generation
            model_id: Model identifier (e.g., "huihui-moe-60b")
        """
        super().__init__(
            name="CreativeExpert",
            domain="creative",
            lumen_primary=Lumen.ONTICUM,
            model_id=model_id or "huihui-moe-60b"
        )
        self.llm_interface = llm_interface

    async def _process_core(
        self,
        query: str,
        context: OntologicalContext,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, str, float]:
        """
        Core creative processing with LLM.

        Strategy:
        1. Generate novel perspectives
        2. Explore conceptual spaces
        3. Blend disparate ideas
        4. Synthesize innovative solutions
        """
        logger.debug(f"{self.name}: Generating creative insights with LLM")

        # Convert context to dict for LLM interface
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "becoming_aspect": context.becoming_aspect,
            "suchness_aspect": context.suchness_aspect,
            "ethical_stakes": context.ethical_stakes,
        }

        # Query LLM as creative expert
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary="onticum",
            query=query,
            context=context_dict,
            temperature=0.9,  # Higher temperature for creative exploration
        )

        logger.info(
            f"{self.name}: LLM creative generation complete",
            extra={
                "confidence": confidence,
                "claim_length": len(claim),
            }
        )

        return claim, rationale, confidence
