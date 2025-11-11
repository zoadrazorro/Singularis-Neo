"""
Reasoning Expert with LLM Integration

Primary Lumen: STRUCTURALE (Form/Rationality)

Specialization:
- Deductive, inductive, abductive reasoning
- Logical consistency checking
- Mathematical/formal reasoning
- Contradiction detection
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface


class ReasoningExpertLLM(Expert):
    """
    Reasoning Expert with LLM integration.

    From ETHICA UNIVERSALIS:
    "Reason is the second kind of knowledge, understanding things
    through common notions and adequate ideas."
    """

    def __init__(
        self,
        llm_interface: ExpertLLMInterface,
        model_id: Optional[str] = None
    ):
        """
        Initialize reasoning expert with LLM.
        
        Args:
            llm_interface: LLM interface for generation
            model_id: Model identifier (e.g., "huihui-moe-60b")
        """
        super().__init__(
            name="ReasoningExpert",
            domain="reasoning",
            lumen_primary=Lumen.STRUCTURALE,
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
        Core reasoning processing with LLM.

        Strategy:
        1. Identify logical structure
        2. Apply deductive/inductive inference
        3. Check for contradictions
        4. Validate conclusions
        """
        logger.debug(f"{self.name}: Analyzing logical structure with LLM")

        # Convert context to dict for LLM interface
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "becoming_aspect": context.becoming_aspect,
            "suchness_aspect": context.suchness_aspect,
            "ethical_stakes": context.ethical_stakes,
        }

        # Query LLM as reasoning expert
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary="structurale",
            query=query,
            context=context_dict,
            temperature=0.3,  # Lower temperature for logical reasoning
        )

        logger.info(
            f"{self.name}: LLM reasoning complete",
            extra={
                "confidence": confidence,
                "claim_length": len(claim),
            }
        )

        return claim, rationale, confidence
