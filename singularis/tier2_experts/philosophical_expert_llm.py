"""
Philosophical Expert with LLM Integration

Primary Lumen: PARTICIPATUM (Consciousness/Awareness/Reflexivity)

Specialization:
- Ontological analysis
- Metaphysical inquiry
- Ethical reasoning
- Phenomenological insight
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface


class PhilosophicalExpertLLM(Expert):
    """
    Philosophical Expert with LLM integration.

    From ETHICA UNIVERSALIS:
    "The third kind of knowledge proceeds from an adequate idea of certain
    attributes of God to the adequate knowledge of the essence of things."
    This is intuitive knowledge - direct philosophical insight.
    """

    def __init__(
        self,
        llm_interface: ExpertLLMInterface,
        model_id: Optional[str] = None
    ):
        """
        Initialize philosophical expert with LLM.
        
        Args:
            llm_interface: LLM interface for generation
            model_id: Model identifier (e.g., "huihui-moe-60b")
        """
        super().__init__(
            name="PhilosophicalExpert",
            domain="philosophy",
            lumen_primary=Lumen.PARTICIPATUM,
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
        Core philosophical processing with LLM.

        Strategy:
        1. Analyze ontological commitments
        2. Explore metaphysical implications
        3. Ground in philosophical tradition
        4. Provide intuitive insight
        """
        logger.debug(f"{self.name}: Conducting philosophical analysis with LLM")

        # Convert context to dict for LLM interface
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "becoming_aspect": context.becoming_aspect,
            "suchness_aspect": context.suchness_aspect,
            "ethical_stakes": context.ethical_stakes,
        }

        # Query LLM as philosophical expert
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary="participatum",
            query=query,
            context=context_dict,
            temperature=0.7,  # Balanced temperature for philosophical depth
        )

        logger.info(
            f"{self.name}: LLM philosophical analysis complete",
            extra={
                "confidence": confidence,
                "claim_length": len(claim),
            }
        )

        return claim, rationale, confidence
