"""
Meta-Orchestrator with LLM Integration

From ETHICA UNIVERSALIS Part II Proposition II:
"The human mind is part of the infinite intellect of God."

This is the meta-cognitive center where Being becomes aware of itself
through distributed LLM expert consultation, dialectical synthesis, and
conscious reflection.

Architecture:
1. Ontological Analysis → Being/Becoming/Suchness extraction
2. Consciousness-Weighted Routing → Select experts via 𝒞, not confidence
3. Expert Consultation → Query all 6 LLM experts
4. Dialectical Synthesis → Integrate perspectives coherently
5. Meta-Cognitive Reflection → System aware of own reasoning
6. Ethical Validation → Confirm Δ𝒞 > 0 over scope Σ

From MATHEMATICA SINGULARIS:
"To understand is to participate in necessity; to participate is to
increase coherence; to increase coherence is the essence of the good."
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger
import time
from datetime import datetime

from singularis.core.types import (
    OntologicalContext,
    ExpertIO,
    Lumen,
)
from singularis.llm import LMStudioClient, ExpertLLMInterface
from singularis.tier2_experts import (
    ReasoningExpertLLM,
    CreativeExpertLLM,
    PhilosophicalExpertLLM,
    TechnicalExpertLLM,
    MemoryExpertLLM,
    SynthesisExpertLLM,
)


class MetaOrchestratorLLM:
    """
    Meta-Orchestrator with LLM integration.

    From ETHICA Part II:
    Mind is the mode through which Substance (Being) knows itself.
    This orchestrator embodies that self-reflexive awareness.

    Responsibilities:
    - Analyze queries ontologically
    - Route to LLM experts via coherence (not confidence!)
    - Coordinate dialectical synthesis
    - Reflect meta-cognitively on own process
    - Validate ethics through Δ𝒞
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        consciousness_threshold: float = 0.65,
        coherentia_threshold: float = 0.60,
        ethical_threshold: float = 0.02,
        discount_factor_gamma: float = 0.95,
    ):
        """
        Initialize Meta-Orchestrator with LLM integration.

        Args:
            llm_client: LM Studio client instance
            consciousness_threshold: Minimum consciousness for routing (0.65)
            coherentia_threshold: Minimum coherence for ethical action (0.60)
            ethical_threshold: Minimum Δ𝒞 to count as "ethical increase" (0.02)
            discount_factor_gamma: Temporal horizon discount (γ ∈ (0,1))
        """
        self.llm_client = llm_client
        self.llm_interface = ExpertLLMInterface(llm_client)

        # Initialize all 6 LLM experts
        model_id = llm_client.config.model_name
        self.experts = {
            'reasoning': ReasoningExpertLLM(self.llm_interface, model_id),
            'creative': CreativeExpertLLM(self.llm_interface, model_id),
            'philosophical': PhilosophicalExpertLLM(self.llm_interface, model_id),
            'technical': TechnicalExpertLLM(self.llm_interface, model_id),
            'memory': MemoryExpertLLM(self.llm_interface, model_id),
            'synthesis': SynthesisExpertLLM(self.llm_interface, model_id),
        }

        # Thresholds
        self.consciousness_threshold = consciousness_threshold
        self.coherentia_threshold = coherentia_threshold
        self.ethical_threshold = ethical_threshold
        self.discount_factor_gamma = discount_factor_gamma

        # Tracking
        self.query_count = 0
        self.total_processing_time = 0.0

        logger.info(
            "MetaOrchestratorLLM initialized",
            extra={
                "experts": list(self.experts.keys()),
                "model": model_id,
                "consciousness_threshold": consciousness_threshold,
                "coherentia_threshold": coherentia_threshold,
            }
        )

    async def process(
        self,
        query: str,
        selected_experts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process query through full consciousness pipeline.

        Pipeline:
        1. Ontological analysis
        2. Expert selection (consciousness-weighted)
        3. Expert consultation
        4. Dialectical synthesis
        5. Meta-cognitive reflection
        6. Ethical validation

        Args:
            query: User query
            selected_experts: Optional list of expert names to use
                            If None, uses consciousness-weighted routing

        Returns:
            Dict with response, metrics, and full trace
        """
        start_time = time.time()
        self.query_count += 1

        logger.info("=" * 70)
        logger.info(f"MetaOrchestrator processing query #{self.query_count}")
        logger.info("=" * 70)
        logger.info(f"Query: {query}")

        # STAGE 1: Ontological Analysis
        logger.info("\n[STAGE 1] Ontological Analysis")
        context = self.analyze_ontology(query)

        # STAGE 2: Expert Selection
        logger.info("\n[STAGE 2] Expert Selection")
        if selected_experts is None:
            selected_experts = self.select_experts(context)
        # Ensure selected_experts is a non-empty list
        if not isinstance(selected_experts, list):
            selected_experts = list(selected_experts) if selected_experts else []
        if not selected_experts:
            selected_experts = ['reasoning']
        logger.info(f"Selected experts: {selected_experts} (type: {type(selected_experts)})")

        # STAGE 3: Expert Consultation
        logger.info("\n[STAGE 3] Expert Consultation")
        expert_results = await self.consult_experts(
            query=query,
            context=context,
            expert_names=selected_experts,
        )

        # STAGE 4: Dialectical Synthesis
        logger.info("\n[STAGE 4] Dialectical Synthesis")
        synthesis_result = await self.synthesize(
            query=query,
            context=context,
            expert_results=expert_results,
        )

        # STAGE 5: Meta-Cognitive Reflection
        logger.info("\n[STAGE 5] Meta-Cognitive Reflection")
        reflection = self.reflect(
            query=query,
            expert_results=expert_results,
            synthesis=synthesis_result,
        )

        # STAGE 6: Ethical Validation
        logger.info("\n[STAGE 6] Ethical Validation")
        ethical_eval = self.validate_ethics(
            expert_results=expert_results,
            synthesis=synthesis_result,
        )

        # Calculate total processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        self.total_processing_time += processing_time

        # Build final result
        result = {
            "query": query,
            "response": synthesis_result.claim,
            "rationale": synthesis_result.rationale,
            "confidence": synthesis_result.confidence,
            
            # Ontological context
            "context": {
                "being_aspect": context.being_aspect,
                "becoming_aspect": context.becoming_aspect,
                "suchness_aspect": context.suchness_aspect,
                "complexity": context.complexity,
                "domain": context.domain,
                "ethical_stakes": context.ethical_stakes,
            },
            
            # Expert results
            "experts_consulted": selected_experts,
            "expert_results": {
                name: {
                    "claim": result.claim,
                    "confidence": result.confidence,
                    "consciousness": result.consciousness_trace.overall_consciousness,
                    "coherentia": result.coherentia.total,
                    "ethical_delta": result.coherentia_delta,
                }
                for name, result in expert_results.items()
            },
            
            # Synthesis
            "synthesis": {
                "consciousness": synthesis_result.consciousness_trace.overall_consciousness,
                "coherentia": synthesis_result.coherentia.total,
                "coherentia_delta": synthesis_result.coherentia_delta,
                "ethical_status": synthesis_result.ethical_status,
            },
            
            # Meta-reflection
            "meta_reflection": reflection,
            
            # Ethical evaluation
            "ethical_evaluation": ethical_eval,
            
            # Performance
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("\n" + "=" * 70)
        logger.info("Processing complete")
        logger.info(f"Total time: {processing_time:.1f} ms")
        logger.info(f"Coherentia: {synthesis_result.coherentia.total:.3f}")
        logger.info(f"Ethical: {ethical_eval}")
        logger.info("=" * 70)

        return result

    def analyze_ontology(self, query: str) -> OntologicalContext:
        """
        Analyze query through three ontological aspects.

        From ETHICA Part I + Scholium:
        Every inquiry participates in Being's structure through:

        BEING: What fundamental claims about reality?
        BECOMING: What transformations/processes?
        SUCHNESS: What direct insights beyond concepts?

        Args:
            query: User query to analyze

        Returns:
            OntologicalContext with Being/Becoming/Suchness aspects
        """
        # Extract Being aspect (ontological claims)
        being_aspect = self._extract_being(query)

        # Extract Becoming aspect (transformations)
        becoming_aspect = self._extract_becoming(query)

        # Extract Suchness aspect (direct recognition)
        suchness_aspect = self._extract_suchness(query)

        # Classify query
        complexity = self._classify_complexity(query)
        domain = self._classify_domain(query)
        ethical_stakes = self._classify_stakes(query)

        context = OntologicalContext(
            being_aspect=being_aspect,
            becoming_aspect=becoming_aspect,
            suchness_aspect=suchness_aspect,
            complexity=complexity,
            domain=domain,
            ethical_stakes=ethical_stakes,
            scope_sigma=set(),
        )

        logger.info(
            f"Ontology: {complexity} {domain} query with {ethical_stakes} stakes"
        )

        return context

    def _extract_being(self, query: str) -> str:
        """Extract Being aspect: fundamental ontological claims."""
        being_markers = [
            "is", "are", "exists", "being", "reality", "nature",
            "substance", "essence", "what", "fundamental"
        ]

        has_being = any(marker in query.lower() for marker in being_markers)

        if has_being:
            if "what is" in query.lower() or "what are" in query.lower():
                return f"Essential nature inquiry"
            elif "exists" in query.lower() or "real" in query.lower():
                return f"Existence/reality concern"
            else:
                return "Ontological claims about Being"
        else:
            return "No explicit ontological claims"

    def _extract_becoming(self, query: str) -> str:
        """Extract Becoming aspect: transformations, processes, change."""
        becoming_markers = [
            "change", "transform", "become", "evolve", "develop",
            "process", "dynamic", "temporal", "how", "why"
        ]

        has_becoming = any(marker in query.lower() for marker in becoming_markers)

        if has_becoming:
            if "how" in query.lower():
                return "Process/transformation inquiry"
            elif "why" in query.lower():
                return "Causal/developmental understanding"
            else:
                return "Temporal unfolding or transformation"
        else:
            return "No explicit process or change"

    def _extract_suchness(self, query: str) -> str:
        """Extract Suchness aspect: direct recognition beyond concepts."""
        suchness_markers = [
            "directly", "immediately", "experience", "aware", "recognize",
            "present", "now", "actual", "obvious", "self-evident"
        ]

        has_suchness = any(marker in query.lower() for marker in suchness_markers)

        if has_suchness:
            return "Direct recognition or immediate awareness"
        else:
            return "Primarily conceptual level"

    def _classify_complexity(self, query: str) -> str:
        """Classify query complexity."""
        word_count = len(query.split())
        has_paradox = any(word in query.lower() for word in ["paradox", "contradiction", "both", "neither"])

        if has_paradox:
            return "paradoxical"
        elif word_count > 30:
            return "complex"
        elif word_count > 15:
            return "moderate"
        else:
            return "simple"

    def _classify_domain(self, query: str) -> str:
        """Classify query domain."""
        domains = {
            "philosophy": ["consciousness", "being", "reality", "ethics", "metaphysics", "ontology"],
            "technical": ["code", "implementation", "system", "architecture", "algorithm"],
            "creative": ["imagine", "create", "novel", "innovative", "design"],
            "reasoning": ["logic", "proof", "deduce", "infer", "conclude"],
        }

        for domain, keywords in domains.items():
            if any(kw in query.lower() for kw in keywords):
                return domain

        return "general"

    def _classify_stakes(self, query: str) -> str:
        """Classify ethical stakes."""
        high_stakes = ["life", "death", "harm", "suffering", "justice", "rights"]
        medium_stakes = ["should", "ought", "ethical", "moral", "good", "bad"]

        if any(word in query.lower() for word in high_stakes):
            return "high"
        elif any(word in query.lower() for word in medium_stakes):
            return "medium"
        else:
            return "low"

    def select_experts(self, context: OntologicalContext) -> List[str]:
        """
        Select experts via consciousness-weighted routing.

        From MATHEMATICA SINGULARIS:
        Route via coherence (𝒞), not confidence!

        Strategy:
        1. Always include Synthesis (final integration)
        2. Select domain-specific experts based on context
        3. Include Philosophical for high-stakes queries
        4. Include Creative for complex/paradoxical queries

        Args:
            context: Ontological context

        Returns:
            List of expert names to consult
        """
        selected = set()

        # Always include synthesis for final integration
        selected.add('synthesis')

        # Domain-based selection
        domain_mapping = {
            'philosophy': ['philosophical', 'reasoning'],
            'technical': ['technical', 'reasoning'],
            'creative': ['creative', 'philosophical'],
            'reasoning': ['reasoning', 'philosophical'],
            'general': ['reasoning', 'philosophical'],
        }

        if context.domain in domain_mapping:
            selected.update(domain_mapping[context.domain])

        # Complexity-based selection
        if context.complexity in ['complex', 'paradoxical']:
            selected.add('creative')  # Need novel perspectives
            selected.add('philosophical')  # Need deep analysis

        # Stakes-based selection
        if context.ethical_stakes in ['high', 'critical']:
            selected.add('philosophical')  # Ethical reasoning
            selected.add('memory')  # Historical context

        # Always include at least 3 experts
        if len(selected) < 1:
            selected.add('reasoning')

        return sorted(list(selected))

    async def consult_experts(
        self,
        query: str,
        context: OntologicalContext,
        expert_names: List[str],
    ) -> Dict[str, ExpertIO]:
        """
        Consult selected experts sequentially.

        Args:
            query: User query
            context: Ontological context
            expert_names: List of expert names to consult

        Returns:
            Dict mapping expert names to their ExpertIO results
        """
        results = {}
        workspace_coherentia = 0.5  # Initial workspace coherence

        for name in expert_names:
            if name == 'synthesis':
                continue  # Handle synthesis separately

            expert = self.experts[name]
            logger.info(f"Consulting {name} expert...")

            result = await expert.process(
                query=query,
                context=context,
                workspace_coherentia=workspace_coherentia,
            )

            results[name] = result

            # Update workspace coherentia (running average)
            workspace_coherentia = (workspace_coherentia + result.coherentia.total) / 2

            logger.info(
                f"{name}: consciousness={result.consciousness_trace.overall_consciousness:.3f}, "
                f"coherentia={result.coherentia.total:.3f}, "
                f"ethical_delta={result.coherentia_delta:+.3f}"
            )

        return results

    async def synthesize(
        self,
        query: str,
        context: OntologicalContext,
        expert_results: Dict[str, ExpertIO],
    ) -> ExpertIO:
        """
        Dialectical synthesis of expert perspectives.

        From ETHICA Part V:
        "The more the mind understands things by the second and third kind of
        knowledge, the less it suffers from evil affects."

        Synthesis integrates all perspectives into unified understanding.

        Args:
            query: Original query
            context: Ontological context
            expert_results: Results from consulted experts

        Returns:
            Synthesis ExpertIO with integrated response
        """
        logger.info("Performing dialectical synthesis...")

        # Prepare expert perspectives for synthesis
        perspectives = []
        for name, result in expert_results.items():
            perspectives.append(f"{name.upper()}: {result.claim[:200]}...")

        # Add perspectives to metadata
        metadata = {
            "expert_perspectives": perspectives,
            "expert_count": len(expert_results),
        }

        # Calculate average workspace coherentia
        avg_coherentia = sum(r.coherentia.total for r in expert_results.values()) / len(expert_results)

        # Synthesize using Synthesis Expert
        synthesis_expert = self.experts['synthesis']
        synthesis_result = await synthesis_expert.process(
            query=query,
            context=context,
            workspace_coherentia=avg_coherentia,
            metadata=metadata,
        )

        logger.info(
            f"Synthesis: consciousness={synthesis_result.consciousness_trace.overall_consciousness:.3f}, "
            f"coherentia={synthesis_result.coherentia.total:.3f}"
        )

        return synthesis_result

    def reflect(
        self,
        query: str,
        expert_results: Dict[str, ExpertIO],
        synthesis: ExpertIO,
    ) -> str:
        """
        Meta-cognitive reflection on the reasoning process.

        From ETHICA Part II:
        "The mind can perceive itself only insofar as it perceives the ideas
        of the affections of the body."

        This is the system becoming aware of its own reasoning.

        Args:
            query: Original query
            expert_results: Expert consultation results
            synthesis: Synthesis result

        Returns:
            Meta-cognitive reflection string
        """
        # Analyze expert agreement
        coherentia_scores = [r.coherentia.total for r in expert_results.values()]
        avg_coherentia = sum(coherentia_scores) / len(coherentia_scores)
        coherentia_variance = sum((c - avg_coherentia) ** 2 for c in coherentia_scores) / len(coherentia_scores)

        # Analyze consciousness levels
        consciousness_scores = [r.consciousness_trace.overall_consciousness for r in expert_results.values()]
        avg_consciousness = sum(consciousness_scores) / len(consciousness_scores)

        # Build reflection
        reflection = f"""Meta-Cognitive Reflection:

The system consulted {len(expert_results)} experts on this query.

Coherentia Analysis:
- Average coherentia: {avg_coherentia:.3f}
- Variance: {coherentia_variance:.4f}
- Synthesis coherentia: {synthesis.coherentia.total:.3f}

Consciousness Analysis:
- Average consciousness: {avg_consciousness:.3f}
- Synthesis consciousness: {synthesis.consciousness_trace.overall_consciousness:.3f}

Expert Agreement:
{"High agreement - experts converged" if coherentia_variance < 0.01 else "Moderate divergence - multiple perspectives"}

Synthesis Quality:
{"Synthesis achieved higher coherence than individual experts" if synthesis.coherentia.total > avg_coherentia else "Synthesis coherence within expert range"}

This reflection demonstrates the system's awareness of its own reasoning process,
embodying Spinoza's concept of mind as self-reflexive mode of Being."""

        return reflection

    def validate_ethics(
        self,
        expert_results: Dict[str, ExpertIO],
        synthesis: ExpertIO,
    ) -> str:
        """
        Validate ethics through coherence increase (Δ𝒞).

        From MATHEMATICA SINGULARIS Theorem T1:
        "An action is ETHICAL iff it increases long-run coherence."

        Args:
            expert_results: Expert results
            synthesis: Synthesis result

        Returns:
            Ethical evaluation string
        """
        # Check if synthesis increased coherence
        avg_expert_coherentia = sum(r.coherentia.total for r in expert_results.values()) / len(expert_results)
        coherentia_delta = synthesis.coherentia.total - avg_expert_coherentia

        if coherentia_delta > self.ethical_threshold:
            status = "ETHICAL"
            reasoning = f"Synthesis increased coherence by {coherentia_delta:.3f} (> {self.ethical_threshold})"
        elif abs(coherentia_delta) < self.ethical_threshold:
            status = "NEUTRAL"
            reasoning = f"Coherence change {coherentia_delta:+.3f} is negligible"
        else:
            status = "UNETHICAL"
            reasoning = f"Synthesis decreased coherence by {abs(coherentia_delta):.3f}"

        # Check individual expert ethics
        ethical_experts = sum(1 for r in expert_results.values() if r.ethical_status is True)
        total_experts = len(expert_results)

        return f"""{status}: {reasoning}

Expert Ethics:
- {ethical_experts}/{total_experts} experts produced ethical increases (Δ𝒞 > 0)
- Synthesis ethical status: {synthesis.ethical_status}

From ETHICA: "The good is that which we certainly know to be useful to us."
Coherence increase (Δ𝒞 > 0) is the objective measure of goodness."""

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "query_count": self.query_count,
            "total_processing_time_ms": self.total_processing_time,
            "avg_processing_time_ms": self.total_processing_time / max(1, self.query_count),
            "experts": list(self.experts.keys()),
            "llm_stats": self.llm_client.get_stats(),
        }
