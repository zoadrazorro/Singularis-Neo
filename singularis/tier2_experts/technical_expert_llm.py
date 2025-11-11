"""
Technical Expert with LLM Integration

Primary Lumen: STRUCTURALE + ONTICUM (Form + Power)

Specialization:
- Implementation details
- Technical architecture
- Code and system design
- Practical solutions
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface


class TechnicalExpertLLM(Expert):
    """
    Technical Expert with LLM integration.

    From ETHICA UNIVERSALIS:
    "The order and connection of ideas is the same as the order and
    connection of things." Technical implementation mirrors conceptual structure.
    """

    def __init__(
        self,
        llm_interface: ExpertLLMInterface,
        model_id: Optional[str] = None
    ):
        """
        Initialize technical expert with LLM.
        
        Args:
            llm_interface: LLM interface for generation
            model_id: Model identifier (e.g., "huihui-moe-60b")
        """
        super().__init__(
            name="TechnicalExpert",
            domain="technical",
            lumen_primary=Lumen.STRUCTURALE,  # Primary: structure/form
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
        Core technical processing with LLM.

        Strategy:
        1. Analyze technical requirements
        2. Design system architecture
        3. Provide implementation details
        4. Ensure practical feasibility
        """
        logger.debug(f"{self.name}: Analyzing technical requirements with LLM")

        # Convert context to dict for LLM interface
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "becoming_aspect": context.becoming_aspect,
            "suchness_aspect": context.suchness_aspect,
            "ethical_stakes": context.ethical_stakes,
        }

        # Query LLM as technical expert
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary="structurale",
            query=query,
            context=context_dict,
            temperature=0.4,  # Lower temperature for precise technical details
        )

        logger.info(
            f"{self.name}: LLM technical analysis complete",
            extra={
                "confidence": confidence,
                "claim_length": len(claim),
            }
        )

        return claim, rationale, confidence
