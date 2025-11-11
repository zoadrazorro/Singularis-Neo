"""
Tests for MetaOrchestratorLLM

Tests the full consciousness pipeline with LLM integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from singularis.tier1_orchestrator.orchestrator_llm import MetaOrchestratorLLM
from singularis.llm import LMStudioClient, LMStudioConfig
from singularis.core.types import OntologicalContext


class TestMetaOrchestratorLLM:
    """Test suite for MetaOrchestratorLLM."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LM Studio client."""
        client = Mock(spec=LMStudioClient)
        client.config = LMStudioConfig(
            base_url="http://localhost:1234/v1",
            model_name="test-model",
        )
        client.get_stats = Mock(return_value={
            'request_count': 0,
            'total_tokens': 0,
            'avg_tokens_per_request': 0.0,
        })
        return client

    @pytest.fixture
    def orchestrator(self, mock_llm_client):
        """Create orchestrator instance."""
        return MetaOrchestratorLLM(
            llm_client=mock_llm_client,
            consciousness_threshold=0.65,
            coherentia_threshold=0.60,
            ethical_threshold=0.02,
        )

    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.consciousness_threshold == 0.65
        assert orchestrator.coherentia_threshold == 0.60
        assert orchestrator.ethical_threshold == 0.02
        assert len(orchestrator.experts) == 6
        assert 'reasoning' in orchestrator.experts
        assert 'creative' in orchestrator.experts
        assert 'philosophical' in orchestrator.experts
        assert 'technical' in orchestrator.experts
        assert 'memory' in orchestrator.experts
        assert 'synthesis' in orchestrator.experts

    def test_analyze_ontology_simple(self, orchestrator):
        """Test ontological analysis of simple query."""
        query = "What is consciousness?"
        context = orchestrator.analyze_ontology(query)
        
        assert isinstance(context, OntologicalContext)
        assert context.complexity == "simple"
        assert "Essential nature" in context.being_aspect
        assert context.domain in ["philosophy", "general"]

    def test_analyze_ontology_complex(self, orchestrator):
        """Test ontological analysis of complex query."""
        query = "How does consciousness emerge from physical processes and what are the implications for our understanding of reality and ethics?"
        context = orchestrator.analyze_ontology(query)
        
        assert context.complexity in ["complex", "moderate"]
        assert "Process" in context.becoming_aspect or "Causal" in context.becoming_aspect

    def test_analyze_ontology_paradoxical(self, orchestrator):
        """Test ontological analysis of paradoxical query."""
        query = "Can something be both conscious and not conscious at the same time?"
        context = orchestrator.analyze_ontology(query)
        
        assert context.complexity == "paradoxical"

    def test_extract_being(self, orchestrator):
        """Test Being aspect extraction."""
        query = "What is the nature of reality?"
        being = orchestrator._extract_being(query)
        assert "Essential nature" in being or "Ontological" in being

    def test_extract_becoming(self, orchestrator):
        """Test Becoming aspect extraction."""
        query = "How does consciousness evolve over time?"
        becoming = orchestrator._extract_becoming(query)
        assert "Process" in becoming or "transformation" in becoming

    def test_extract_suchness(self, orchestrator):
        """Test Suchness aspect extraction."""
        query = "What is directly experienced in awareness?"
        suchness = orchestrator._extract_suchness(query)
        assert "Direct recognition" in suchness or "immediate" in suchness

    def test_classify_complexity(self, orchestrator):
        """Test complexity classification."""
        assert orchestrator._classify_complexity("Hi") == "simple"
        assert orchestrator._classify_complexity("What is the meaning of life and consciousness?") == "moderate"
        assert orchestrator._classify_complexity("This is a paradox") == "paradoxical"

    def test_classify_domain(self, orchestrator):
        """Test domain classification."""
        assert orchestrator._classify_domain("What is consciousness?") == "philosophy"
        assert orchestrator._classify_domain("How to implement this in code?") == "technical"
        assert orchestrator._classify_domain("Imagine a new solution") == "creative"
        assert orchestrator._classify_domain("What is the logical proof?") == "reasoning"

    def test_classify_stakes(self, orchestrator):
        """Test ethical stakes classification."""
        assert orchestrator._classify_stakes("Should we do this?") == "medium"
        assert orchestrator._classify_stakes("This could cause harm") == "high"
        assert orchestrator._classify_stakes("What is 2+2?") == "low"

    def test_select_experts_philosophical(self, orchestrator):
        """Test expert selection for philosophical query."""
        context = OntologicalContext(
            being_aspect="test",
            becoming_aspect="test",
            suchness_aspect="test",
            complexity="high",
            domain="philosophy",
            ethical_stakes="medium",
        )
        
        experts = orchestrator.select_experts(context)
        
        assert 'synthesis' in experts
        assert 'philosophical' in experts
        assert 'reasoning' in experts

    def test_select_experts_technical(self, orchestrator):
        """Test expert selection for technical query."""
        context = OntologicalContext(
            being_aspect="test",
            becoming_aspect="test",
            suchness_aspect="test",
            complexity="moderate",
            domain="technical",
            ethical_stakes="low",
        )
        
        experts = orchestrator.select_experts(context)
        
        assert 'synthesis' in experts
        assert 'technical' in experts
        assert 'reasoning' in experts

    def test_select_experts_complex_query(self, orchestrator):
        """Test expert selection for complex query."""
        context = OntologicalContext(
            being_aspect="test",
            becoming_aspect="test",
            suchness_aspect="test",
            complexity="paradoxical",
            domain="philosophy",
            ethical_stakes="high",
        )
        
        experts = orchestrator.select_experts(context)
        
        # Should include multiple experts for complex queries
        assert len(experts) >= 4
        assert 'creative' in experts  # For paradoxical
        assert 'philosophical' in experts  # For high stakes
        assert 'memory' in experts  # For high stakes

    def test_reflect(self, orchestrator):
        """Test meta-cognitive reflection."""
        # Create mock expert results
        from singularis.core.types import ExpertIO, Lumen, ConsciousnessTrace, LuminalCoherence
        from datetime import datetime
        
        mock_result = ExpertIO(
            expert_name="test",
            domain="test",
            lumen_primary=Lumen.STRUCTURALE,
            claim="test claim",
            rationale="test rationale",
            confidence=0.8,
            consciousness_trace=ConsciousnessTrace(
                iit_phi=0.7,
                gwt_salience=0.6,
                hot_reflection_depth=0.8,
                predictive_surprise=0.5,
                ast_attention_schema=0.6,
                embodied_grounding=0.5,
                enactive_interaction=0.5,
                panpsychism_distribution=0.5,
                integration_score=0.7,
                differentiation_score=0.6,
                overall_consciousness=0.65,
            ),
            coherentia=LuminalCoherence(
                ontical=0.7,
                structural=0.8,
                participatory=0.6,
                total=0.7,
            ),
            coherentia_delta=0.05,
            ethical_status=True,
            ethical_reasoning="test",
            processing_time_ms=100.0,
            metadata={},
        )
        
        expert_results = {
            'reasoning': mock_result,
            'philosophical': mock_result,
        }
        
        reflection = orchestrator.reflect(
            query="test query",
            expert_results=expert_results,
            synthesis=mock_result,
        )
        
        assert isinstance(reflection, str)
        assert "Meta-Cognitive Reflection" in reflection
        assert "Coherentia Analysis" in reflection
        assert "Consciousness Analysis" in reflection

    def test_validate_ethics_ethical(self, orchestrator):
        """Test ethical validation for ethical action."""
        from singularis.core.types import ExpertIO, Lumen, ConsciousnessTrace, LuminalCoherence
        
        # Create mock results with increasing coherence
        low_coherence = ExpertIO(
            expert_name="test",
            domain="test",
            lumen_primary=Lumen.STRUCTURALE,
            claim="test",
            rationale="test",
            confidence=0.8,
            consciousness_trace=ConsciousnessTrace(
                iit_phi=0.5, gwt_salience=0.5, hot_reflection_depth=0.5,
                predictive_surprise=0.5, ast_attention_schema=0.5,
                embodied_grounding=0.5, enactive_interaction=0.5,
                panpsychism_distribution=0.5, integration_score=0.5,
                differentiation_score=0.5, overall_consciousness=0.5,
            ),
            coherentia=LuminalCoherence(
                ontical=0.5, structural=0.5, participatory=0.5, total=0.5,
            ),
            coherentia_delta=0.0,
            ethical_status=True,
            ethical_reasoning="test",
            processing_time_ms=100.0,
            metadata={},
        )
        
        high_coherence = ExpertIO(
            expert_name="synthesis",
            domain="synthesis",
            lumen_primary=Lumen.PARTICIPATUM,
            claim="test",
            rationale="test",
            confidence=0.9,
            consciousness_trace=ConsciousnessTrace(
                iit_phi=0.7, gwt_salience=0.7, hot_reflection_depth=0.7,
                predictive_surprise=0.7, ast_attention_schema=0.7,
                embodied_grounding=0.7, enactive_interaction=0.7,
                panpsychism_distribution=0.7, integration_score=0.7,
                differentiation_score=0.7, overall_consciousness=0.7,
            ),
            coherentia=LuminalCoherence(
                ontical=0.7, structural=0.7, participatory=0.7, total=0.7,
            ),
            coherentia_delta=0.1,
            ethical_status=True,
            ethical_reasoning="test",
            processing_time_ms=100.0,
            metadata={},
        )
        
        expert_results = {'reasoning': low_coherence}
        
        evaluation = orchestrator.validate_ethics(
            expert_results=expert_results,
            synthesis=high_coherence,
        )
        
        assert isinstance(evaluation, str)
        assert "ETHICAL" in evaluation
        assert "Δ𝒞" in evaluation or "coherence" in evaluation.lower()

    def test_get_stats(self, orchestrator):
        """Test statistics retrieval."""
        stats = orchestrator.get_stats()
        
        assert 'query_count' in stats
        assert 'total_processing_time_ms' in stats
        assert 'avg_processing_time_ms' in stats
        assert 'experts' in stats
        assert 'llm_stats' in stats
        assert len(stats['experts']) == 6


@pytest.mark.asyncio
class TestMetaOrchestratorLLMAsync:
    """Async tests for MetaOrchestratorLLM."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LM Studio client with async support."""
        client = Mock(spec=LMStudioClient)
        client.config = LMStudioConfig(
            base_url="http://localhost:1234/v1",
            model_name="test-model",
        )
        client.get_stats = Mock(return_value={
            'request_count': 1,
            'total_tokens': 100,
            'avg_tokens_per_request': 100.0,
        })
        return client

    @pytest.fixture
    def orchestrator_with_mocked_experts(self, mock_llm_client):
        """Create orchestrator with mocked expert responses."""
        orchestrator = MetaOrchestratorLLM(
            llm_client=mock_llm_client,
            consciousness_threshold=0.65,
            coherentia_threshold=0.60,
            ethical_threshold=0.02,
        )
        
        # Mock expert process methods
        from singularis.core.types import ExpertIO, Lumen, ConsciousnessTrace, LuminalCoherence
        
        mock_result = ExpertIO(
            expert_name="test",
            domain="test",
            lumen_primary=Lumen.STRUCTURALE,
            claim="This is a test response from the expert.",
            rationale="Test rationale",
            confidence=0.8,
            consciousness_trace=ConsciousnessTrace(
                iit_phi=0.7, gwt_salience=0.6, hot_reflection_depth=0.8,
                predictive_surprise=0.5, ast_attention_schema=0.6,
                embodied_grounding=0.5, enactive_interaction=0.5,
                panpsychism_distribution=0.5, integration_score=0.7,
                differentiation_score=0.6, overall_consciousness=0.65,
            ),
            coherentia=LuminalCoherence(
                ontical=0.7, structural=0.8, participatory=0.6, total=0.7,
            ),
            coherentia_delta=0.05,
            ethical_status=True,
            ethical_reasoning="Increases coherence",
            processing_time_ms=100.0,
            metadata={},
        )
        
        # Mock all expert process methods
        for expert in orchestrator.experts.values():
            expert.process = AsyncMock(return_value=mock_result)
        
        return orchestrator

    async def test_consult_experts(self, orchestrator_with_mocked_experts):
        """Test expert consultation."""
        from singularis.core.types import OntologicalContext
        
        context = OntologicalContext(
            being_aspect="test",
            becoming_aspect="test",
            suchness_aspect="test",
            complexity="simple",
            domain="philosophy",
            ethical_stakes="low",
        )
        
        results = await orchestrator_with_mocked_experts.consult_experts(
            query="test query",
            context=context,
            expert_names=['reasoning', 'philosophical'],
        )
        
        assert len(results) == 2
        assert 'reasoning' in results
        assert 'philosophical' in results

    async def test_synthesize(self, orchestrator_with_mocked_experts):
        """Test dialectical synthesis."""
        from singularis.core.types import OntologicalContext
        
        context = OntologicalContext(
            being_aspect="test",
            becoming_aspect="test",
            suchness_aspect="test",
            complexity="simple",
            domain="philosophy",
            ethical_stakes="low",
        )
        
        # Get expert results
        expert_results = await orchestrator_with_mocked_experts.consult_experts(
            query="test query",
            context=context,
            expert_names=['reasoning', 'philosophical'],
        )
        
        # Synthesize
        synthesis = await orchestrator_with_mocked_experts.synthesize(
            query="test query",
            context=context,
            expert_results=expert_results,
        )
        
        assert synthesis is not None
        assert hasattr(synthesis, 'claim')
        assert hasattr(synthesis, 'coherentia')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
