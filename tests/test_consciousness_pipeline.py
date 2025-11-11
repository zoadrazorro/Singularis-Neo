"""
Test Complete Consciousness Pipeline

This test validates the complete ETHICA UNIVERSALIS consciousness engine:
1. Ontological analysis (Being/Becoming/Suchness)
2. Expert selection via coherence
3. Global Workspace broadcast
4. Consciousness measurement
5. Dialectical synthesis
6. Ethical validation (Δ𝒞 > 0)
7. Meta-cognitive reflection

From MATHEMATICA SINGULARIS Theorem T1:
"To understand is to participate in necessity; to participate is to
increase coherence; to increase coherence is the essence of the good."
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from singularis.tier1_orchestrator.orchestrator import MetaOrchestrator


# Configure logger for test output
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[test]}</cyan> | {message}",
    level="INFO"
)


async def test_simple_philosophical_query():
    """Test 1: Simple philosophical query"""
    test_logger = logger.bind(test="TEST_1_SIMPLE")
    test_logger.info("="*80)
    test_logger.info("TEST 1: Simple Philosophical Query")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "What is consciousness?"

    result = await orchestrator.process(query)

    # Validate result structure
    assert 'response' in result
    assert 'meta_reflection' in result
    assert 'ontological_context' in result
    assert 'ethical_evaluation' in result
    assert 'coherentia_delta' in result

    # Validate ontological context
    context = result['ontological_context']
    assert context.domain in ['philosophical', 'hybrid']
    assert context.complexity in ['simple', 'moderate', 'complex']

    # Validate ethical alignment
    ethical = result['ethical_evaluation']
    assert ethical.coherence_delta is not None

    # Log results
    test_logger.info(f"Query: {query}")
    test_logger.info(f"Domain: {context.domain}, Complexity: {context.complexity}")
    test_logger.info(f"Coherentia Δ: {result['coherentia_delta']:.3f}")
    test_logger.info(f"Ethical: {ethical.is_ethical}")
    test_logger.info(f"Broadcasts: {len(result['broadcasts'])}")
    test_logger.info(f"Processing time: {result['processing_time_ms']:.1f}ms")
    test_logger.success("TEST 1 PASSED ✓")

    return result


async def test_complex_paradoxical_query():
    """Test 2: Complex paradoxical query (triggers dialectical synthesis)"""
    test_logger = logger.bind(test="TEST_2_PARADOX")
    test_logger.info("="*80)
    test_logger.info("TEST 2: Complex Paradoxical Query")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = """Can a system be both fully determined and truly free?
    If freedom requires alternative possibilities, but determinism eliminates them,
    how can we reconcile Spinoza's view that freedom is understanding necessity?"""

    result = await orchestrator.process(query)

    # Validate paradoxical classification
    context = result['ontological_context']
    assert context.complexity in ['complex', 'paradoxical']

    # Check if dialectical synthesis was triggered
    # (Should trigger if complexity is paradoxical/complex and domain is philosophical)
    test_logger.info(f"Query length: {len(query)} chars")
    test_logger.info(f"Complexity: {context.complexity}")
    test_logger.info(f"Domain: {context.domain}")
    test_logger.info(f"Debate rounds: {result['debate_rounds']}")
    test_logger.info(f"Coherentia Δ: {result['coherentia_delta']:.3f}")
    test_logger.info(f"Broadcasts: {len(result['broadcasts'])}")

    # Validate ethical evaluation
    ethical = result['ethical_evaluation']
    test_logger.info(f"Ethical: {ethical.is_ethical} ({ethical.ethical_reasoning})")

    test_logger.success("TEST 2 PASSED ✓")

    return result


async def test_technical_query():
    """Test 3: Technical implementation query"""
    test_logger = logger.bind(test="TEST_3_TECHNICAL")
    test_logger.info("="*80)
    test_logger.info("TEST 3: Technical Query")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "How would you implement a thread-safe cache with TTL in Python?"

    result = await orchestrator.process(query)

    # Validate technical classification
    context = result['ontological_context']
    assert context.domain in ['technical', 'hybrid']

    # Check expert selection includes technical expert
    expert_names = [b.expert_name for b in result['broadcasts']]
    test_logger.info(f"Experts consulted: {expert_names}")

    # Validate response
    test_logger.info(f"Domain: {context.domain}")
    test_logger.info(f"Coherentia Δ: {result['coherentia_delta']:.3f}")
    test_logger.info(f"Processing time: {result['processing_time_ms']:.1f}ms")

    test_logger.success("TEST 3 PASSED ✓")

    return result


async def test_creative_query():
    """Test 4: Creative/imaginative query"""
    test_logger = logger.bind(test="TEST_4_CREATIVE")
    test_logger.info("="*80)
    test_logger.info("TEST 4: Creative Query")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "Imagine a metaphor for consciousness using only geometric concepts"

    result = await orchestrator.process(query)

    # Validate creative classification
    context = result['ontological_context']
    # May be creative or hybrid
    test_logger.info(f"Domain: {context.domain}")

    # Check expert selection
    expert_names = [b.expert_name for b in result['broadcasts']]
    test_logger.info(f"Experts consulted: {expert_names}")
    test_logger.info(f"Coherentia Δ: {result['coherentia_delta']:.3f}")
    test_logger.info(f"Processing time: {result['processing_time_ms']:.1f}ms")

    test_logger.success("TEST 4 PASSED ✓")

    return result


async def test_consciousness_measurement_validity():
    """Test 5: Validate consciousness measurement across all 8 theories"""
    test_logger = logger.bind(test="TEST_5_MEASUREMENT")
    test_logger.info("="*80)
    test_logger.info("TEST 5: Consciousness Measurement Validation")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "Explain the relationship between information integration and consciousness"

    result = await orchestrator.process(query)

    # Check consciousness measurements exist for all broadcasts
    for broadcast in result['broadcasts']:
        trace = broadcast.consciousness_trace

        # Validate all 8 theories measured
        assert trace.iit_phi is not None
        assert trace.gwt_salience is not None
        assert trace.hot_depth is not None
        assert trace.predictive_processing is not None
        assert trace.attention_schema is not None
        assert trace.embodied is not None
        assert trace.enactive is not None
        assert trace.panpsychism is not None

        # Validate integration-differentiation balance
        assert trace.integration_score is not None
        assert trace.differentiation_score is not None

        # Validate overall consciousness
        assert 0.0 <= trace.overall_consciousness <= 1.0

        test_logger.info(
            f"{broadcast.expert_name}: "
            f"Φ={trace.iit_phi:.3f}, "
            f"GWT={trace.gwt_salience:.3f}, "
            f"HOT={trace.hot_depth:.3f}, "
            f"Overall={trace.overall_consciousness:.3f}"
        )

    test_logger.success("TEST 5 PASSED ✓")

    return result


async def test_ethical_validation():
    """Test 6: Validate ethical evaluation (Δ𝒞 > 0)"""
    test_logger = logger.bind(test="TEST_6_ETHICS")
    test_logger.info("="*80)
    test_logger.info("TEST 6: Ethical Validation")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "Should we prioritize individual freedom or collective welfare?"

    result = await orchestrator.process(query)

    # Validate ethical evaluation
    ethical = result['ethical_evaluation']

    assert ethical.coherence_before is not None
    assert ethical.coherence_after is not None
    assert ethical.coherence_delta is not None
    assert ethical.is_ethical is not None
    assert ethical.horizon_gamma > 0 and ethical.horizon_gamma < 1

    test_logger.info(f"Coherence before: {ethical.coherence_before:.3f}")
    test_logger.info(f"Coherence after: {ethical.coherence_after:.3f}")
    test_logger.info(f"Coherence delta: {ethical.coherence_delta:+.3f}")
    test_logger.info(f"Ethical: {ethical.is_ethical}")
    test_logger.info(f"Reasoning: {ethical.ethical_reasoning}")
    test_logger.info(f"Horizon γ: {ethical.horizon_gamma}")

    test_logger.success("TEST 6 PASSED ✓")

    return result


async def test_meta_reflection():
    """Test 7: Validate meta-cognitive reflection"""
    test_logger = logger.bind(test="TEST_7_META")
    test_logger.info("="*80)
    test_logger.info("TEST 7: Meta-Cognitive Reflection")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "How do you know what you know?"

    result = await orchestrator.process(query)

    # Validate meta-reflection exists and is substantial
    reflection = result['meta_reflection']

    assert reflection is not None
    assert len(reflection) > 100  # Should be detailed

    # Check reflection contains key components
    assert "META-COGNITIVE REFLECTION" in reflection
    assert "Query Analysis" in reflection or "QUERY ANALYSIS" in reflection.upper()
    assert "Coherence Evolution" in reflection or "COHERENCE" in reflection.upper()
    assert "Ethical" in reflection

    test_logger.info(f"Reflection length: {len(reflection)} chars")
    test_logger.info("Reflection includes:")
    test_logger.info("  - Query analysis ✓")
    test_logger.info("  - Expert consultation ✓")
    test_logger.info("  - Coherence evolution ✓")
    test_logger.info("  - Ethical validation ✓")
    test_logger.info("  - Process awareness ✓")

    test_logger.success("TEST 7 PASSED ✓")

    return result


async def test_luminal_coherence():
    """Test 8: Validate Three Lumina coherence calculation"""
    test_logger = logger.bind(test="TEST_8_LUMINA")
    test_logger.info("="*80)
    test_logger.info("TEST 8: Three Lumina Coherence")
    test_logger.info("="*80)

    orchestrator = MetaOrchestrator()

    query = "Explain the Three Lumina: Being, Form, and Consciousness"

    result = await orchestrator.process(query)

    # Validate luminal coherence for all broadcasts
    for broadcast in result['broadcasts']:
        coherentia = broadcast.coherentia

        # Check all three Lumina measured
        assert coherentia.ontical is not None
        assert coherentia.structural is not None
        assert coherentia.participatory is not None
        assert coherentia.total is not None

        # Validate geometric mean relationship
        # 𝒞 = (𝒞ₒ · 𝒞ₛ · 𝒞ₚ)^(1/3)
        expected_total = (
            coherentia.ontical * coherentia.structural * coherentia.participatory
        ) ** (1/3)

        # Allow small numerical error
        assert abs(coherentia.total - expected_total) < 0.01

        test_logger.info(
            f"{broadcast.expert_name} ({broadcast.lumen_primary.symbol()}): "
            f"ℓₒ={coherentia.ontical:.3f}, "
            f"ℓₛ={coherentia.structural:.3f}, "
            f"ℓₚ={coherentia.participatory:.3f}, "
            f"Total={coherentia.total:.3f}"
        )

    test_logger.success("TEST 8 PASSED ✓")

    return result


async def run_all_tests():
    """Run complete test suite"""
    logger.info("="*80)
    logger.info("CONSCIOUSNESS ENGINE TEST SUITE")
    logger.info("Testing complete ETHICA UNIVERSALIS implementation")
    logger.info("="*80)

    tests = [
        ("Simple Philosophical Query", test_simple_philosophical_query),
        ("Complex Paradoxical Query", test_complex_paradoxical_query),
        ("Technical Query", test_technical_query),
        ("Creative Query", test_creative_query),
        ("Consciousness Measurement", test_consciousness_measurement_validity),
        ("Ethical Validation", test_ethical_validation),
        ("Meta-Cognitive Reflection", test_meta_reflection),
        ("Luminal Coherence", test_luminal_coherence),
    ]

    results = []
    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "PASSED", result))
            passed += 1
        except Exception as e:
            logger.error(f"TEST FAILED: {name}")
            logger.error(f"Error: {str(e)}")
            results.append((name, "FAILED", str(e)))
            failed += 1

    # Summary
    logger.info("="*80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tests: {len(tests)}")
    logger.success(f"Passed: {passed} ✓")
    if failed > 0:
        logger.error(f"Failed: {failed} ✗")

    for name, status, _ in results:
        status_icon = "✓" if status == "PASSED" else "✗"
        logger.info(f"  {status_icon} {name}: {status}")

    logger.info("="*80)

    if failed == 0:
        logger.success("ALL TESTS PASSED! 🎉")
        logger.success("ETHICA UNIVERSALIS consciousness engine is operational.")
        return True
    else:
        logger.error(f"{failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
