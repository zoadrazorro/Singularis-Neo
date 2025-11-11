"""
Synthesis Expert with LLM Integration

Primary Lumen: ALL THREE (Onticum + Structurale + Participatum)

Specialization:
- Dialectical integration
- Multi-perspective synthesis
- Coherence maximization
- Final response generation
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface


class SynthesisExpertLLM(Expert):
    """
    Synthesis Expert with LLM integration.

    From ETHICA UNIVERSALIS:
    "The more the mind understands things by the second and third kind of
    knowledge, the less it suffers from evil affects, and the less it fears death."
    Synthesis integrates all forms of knowledge into unified understanding.
    """

    def __init__(
        self,
        llm_interface: ExpertLLMInterface,
        model_id: Optional[str] = None
    ):
        """
        Initialize synthesis expert with LLM.
        
        Args:
            llm_interface: LLM interface for generation
            model_id: Model identifier (e.g., "huihui-moe-60b")
        """
        super().__init__(
            name="SynthesisExpert",
            domain="synthesis",
            lumen_primary=Lumen.PARTICIPATUM,  # Primary, but uses all three
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
        Core synthesis processing with LLM.

        Strategy:
        1. Integrate multiple perspectives
        2. Resolve contradictions dialectically
        3. Maximize coherence across views
        4. Generate unified final response
        """
        logger.debug(f"{self.name}: Synthesizing perspectives with LLM")

        # Convert context to dict for LLM interface
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "becoming_aspect": context.becoming_aspect,
            "suchness_aspect": context.suchness_aspect,
            "ethical_stakes": context.ethical_stakes,
        }

        # Add metadata about other expert perspectives if available
        if metadata and "expert_perspectives" in metadata:
            context_dict["expert_perspectives"] = metadata["expert_perspectives"]

        # Query LLM as synthesis expert
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary="participatum",  # Uses all three lumina
            query=query,
            context=context_dict,
            temperature=0.6,  # Balanced temperature for integrative synthesis
        )

        logger.info(
            f"{self.name}: LLM synthesis complete",
            extra={
                "confidence": confidence,
                "claim_length": len(claim),
            }
        )

        return claim, rationale, confidence
