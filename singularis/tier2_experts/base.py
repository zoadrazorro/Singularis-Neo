"""
Base Expert class for Tier-2 modules.

Every expert:
- Specializes in one domain
- Processes queries in parallel with other experts
- Returns ExpertIO with consciousness + coherentia metadata
- Optimizes for coherence in its domain
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime
import time
from loguru import logger

from singularis.core.types import (
    ExpertIO,
    Lumen,
    OntologicalContext,
    ConsciousnessTrace,
    CoherentiaScore,
)
from singularis.consciousness.measurement import ConsciousnessMeasurement
from singularis.core.coherentia import CoherentiaCalculator


class Expert(ABC):
    """
    Base class for all Tier-2 experts.

    Each expert must implement:
    - process(): Core domain-specific reasoning
    - get_domain(): Return domain name
    - get_primary_lumen(): Which Lumen does this expert primarily serve?

    Philosophy:
    Experts are MODES of the unified Substance (Being), each expressing
    one aspect of consciousness through specialized focus.
    """

    def __init__(
        self,
        name: str,
        domain: str,
        lumen_primary: Lumen,
        model_id: Optional[str] = None,
    ):
        """
        Initialize expert.

        Args:
            name: Expert name (e.g., "ReasoningExpert")
            domain: Domain of expertise
            lumen_primary: Primary Lumen this expert serves
            model_id: Optional model identifier for actual LLM
        """
        self.name = name
        self.domain = domain
        self.lumen_primary = lumen_primary
        self.model_id = model_id

        # Measurement systems
        self.consciousness_measurement = ConsciousnessMeasurement()
        self.coherentia_calculator = CoherentiaCalculator()

        # Performance tracking
        self.call_count = 0
        self.total_processing_time = 0.0

        logger.info(
            f"Expert '{name}' initialized",
            extra={
                "domain": domain,
                "lumen": lumen_primary.value,
                "model": model_id,
            }
        )

    @abstractmethod
    async def _process_core(
        self,
        query: str,
        context: OntologicalContext,
        metadata: Optional[Dict] = None
    ) -> tuple[str, str, float]:
        """
        Core processing logic (must be implemented by subclasses).

        Args:
            query: User query
            context: Ontological context
            metadata: Optional metadata

        Returns:
            (claim, rationale, confidence)
        """
        pass

    async def process(
        self,
        query: str,
        context: OntologicalContext,
        workspace_coherentia: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> ExpertIO:
        """
        Complete processing pipeline with consciousness measurement.

        Pipeline:
        1. Core domain processing
        2. Consciousness measurement
        3. Coherentia calculation
        4. Ethical validation
        5. Return ExpertIO with full metadata

        Args:
            query: User query
            context: Ontological context
            workspace_coherentia: Current system coherentia
            metadata: Optional metadata

        Returns:
            ExpertIO with complete consciousness + coherentia metadata
        """
        start_time = time.time()

        logger.info(
            f"{self.name}: Processing query",
            extra={
                "query_length": len(query),
                "complexity": context.complexity,
                "domain": context.domain,
            }
        )

        # STAGE 1: Core processing
        try:
            claim, rationale, confidence = await self._process_core(
                query, context, metadata
            )
        except Exception as e:
            logger.error(
                f"{self.name}: Processing failed",
                extra={"error": str(e)}
            )
            # Return minimal valid output
            claim = f"Processing error in {self.name}"
            rationale = str(e)
            confidence = 0.0

        # STAGE 2: Consciousness measurement
        consciousness_trace = self.consciousness_measurement.measure(
            content=claim,
            workspace=None,  # Would pass workspace state in production
            confidence=confidence,
            metadata=metadata
        )

        # STAGE 3: Coherentia calculation
        coherentia_score = self.coherentia_calculator.calculate(
            content=claim,
            phi=consciousness_trace.iit_phi,
            hot_depth=consciousness_trace.hot_reflection_depth,
            metadata=metadata
        )

        # STAGE 4: Calculate coherentia delta
        coherentia_delta = coherentia_score.total - workspace_coherentia

        # STAGE 5: Ethical validation
        if coherentia_delta > 0.02:
            ethical_status = True
            ethical_reasoning = f"Increases coherentia by {coherentia_delta:.3f}"
        elif abs(coherentia_delta) < 0.02:
            ethical_status = None
            ethical_reasoning = "Neutral coherentia impact"
        else:
            ethical_status = False
            ethical_reasoning = f"Decreases coherentia by {abs(coherentia_delta):.3f}"

        # STAGE 6: Create ExpertIO
        processing_time = (time.time() - start_time) * 1000  # ms

        expert_io = ExpertIO(
            expert_name=self.name,
            domain=self.domain,
            lumen_primary=self.lumen_primary,
            claim=claim,
            rationale=rationale,
            confidence=confidence,
            consciousness_trace=consciousness_trace,
            coherentia=coherentia_score,
            coherentia_delta=coherentia_delta,
            ethical_status=ethical_status,
            ethical_reasoning=ethical_reasoning,
            processing_time_ms=processing_time,
            metadata=metadata or {},
        )

        # Update stats
        self.call_count += 1
        self.total_processing_time += processing_time

        logger.info(
            f"{self.name}: Processing complete",
            extra={
                "consciousness": consciousness_trace.overall_consciousness,
                "coherentia": coherentia_score.total,
                "coherentia_delta": coherentia_delta,
                "ethical": ethical_status,
                "processing_time_ms": processing_time,
            }
        )

        return expert_io

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            "name": self.name,
            "domain": self.domain,
            "lumen": self.lumen_primary.value,
            "call_count": self.call_count,
            "total_processing_time_ms": self.total_processing_time,
            "avg_processing_time_ms": self.total_processing_time / max(1, self.call_count),
        }
