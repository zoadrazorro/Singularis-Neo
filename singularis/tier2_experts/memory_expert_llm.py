"""
Memory Expert with LLM Integration

Primary Lumen: PARTICIPATUM + STRUCTURALE (Consciousness + Form)

Specialization:
- Contextual grounding
- Historical knowledge
- Pattern recognition
- Associative recall
"""

from typing import Dict, Optional, Tuple
from loguru import logger

from singularis.tier2_experts.base import Expert
from singularis.core.types import Lumen, OntologicalContext
from singularis.llm import ExpertLLMInterface


class MemoryExpertLLM(Expert):
    """
    Memory Expert with LLM integration.

    From ETHICA UNIVERSALIS:
    "Memory is nothing but a certain concatenation of ideas involving the
    nature of things which are outside the human body, which concatenation
    occurs in the mind according to the order and concatenation of the
    affections of the human body."
    """

    def __init__(
        self,
        llm_interface: ExpertLLMInterface,
        model_id: Optional[str] = None
    ):
        """
        Initialize memory expert with LLM.
        
        Args:
            llm_interface: LLM interface for generation
            model_id: Model identifier (e.g., "huihui-moe-60b")
        """
        super().__init__(
            name="MemoryExpert",
            domain="memory",
            lumen_primary=Lumen.PARTICIPATUM,  # Primary: consciousness/awareness
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
        Core memory processing with LLM.

        Strategy:
        1. Retrieve relevant context
        2. Identify historical patterns
        3. Ground in prior knowledge
        4. Provide associative connections
        """
        logger.debug(f"{self.name}: Retrieving contextual knowledge with LLM")

        # Convert context to dict for LLM interface
        context_dict = {
            "domain": context.domain,
            "complexity": context.complexity,
            "being_aspect": context.being_aspect,
            "becoming_aspect": context.becoming_aspect,
            "suchness_aspect": context.suchness_aspect,
            "ethical_stakes": context.ethical_stakes,
        }

        # Query LLM as memory expert
        claim, rationale, confidence = await self.llm_interface.expert_query(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary="participatum",
            query=query,
            context=context_dict,
            temperature=0.5,  # Moderate temperature for contextual recall
        )

        logger.info(
            f"{self.name}: LLM memory retrieval complete",
            extra={
                "confidence": confidence,
                "claim_length": len(claim),
            }
        )

        return claim, rationale, confidence
