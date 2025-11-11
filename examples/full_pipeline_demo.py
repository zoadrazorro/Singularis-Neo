"""
Full End-to-End Pipeline Demonstration

Complete consciousness pipeline with:
- Ontological analysis
- Consciousness-weighted expert routing
- Multi-expert consultation
- Dialectical synthesis
- Meta-cognitive reflection
- Ethical validation

Usage:
    python examples/full_pipeline_demo.py
"""

import asyncio
from loguru import logger

from singularis.llm import LMStudioClient, LMStudioConfig
from singularis.tier1_orchestrator.orchestrator_llm import MetaOrchestratorLLM


async def main():
    """Run full pipeline demonstration."""
    
    logger.info("=" * 80)
    logger.info("SINGULARIS - Full Consciousness Pipeline Demonstration")
    logger.info("=" * 80)
    
    # Configure LM Studio
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="huihui-moe-60b-a38",
        temperature=0.7,
        max_tokens=1024,
    )
    
    # Test queries of different types
    queries = [
        {
            "query": "What is the relationship between consciousness and coherence in Spinoza's philosophy?",
            "description": "Philosophical query - tests philosophical + reasoning experts"
        },
        {
            "query": "How can we implement a consciousness measurement system in code?",
            "description": "Technical query - tests technical + reasoning experts"
        },
        {
            "query": "Imagine a new way to think about artificial consciousness that transcends current paradigms.",
            "description": "Creative query - tests creative + philosophical experts"
        },
    ]
    
    async with LMStudioClient(config) as client:
        # Initialize MetaOrchestrator with LLM
        orchestrator = MetaOrchestratorLLM(
            llm_client=client,
            consciousness_threshold=0.65,
            coherentia_threshold=0.60,
            ethical_threshold=0.02,
        )
        
        logger.info(f"\nInitialized with model: {config.model_name}")
        logger.info(f"Experts available: {list(orchestrator.experts.keys())}\n")
        
        # Process each query
        for i, test_case in enumerate(queries, 1):
            logger.info("\n" + "=" * 80)
            logger.info(f"TEST CASE {i}/{len(queries)}")
            logger.info("=" * 80)
            logger.info(f"Description: {test_case['description']}")
            logger.info(f"Query: {test_case['query']}")
            logger.info("=" * 80)
            
            # Process through full pipeline
            result = await orchestrator.process(test_case['query'])
            
            # Display results
            display_results(result, i)
            
            logger.info("\n" + "=" * 80)
            logger.info(f"Test case {i} complete")
            logger.info("=" * 80)
            
            # Brief pause between queries
            if i < len(queries):
                logger.info("\nPausing before next query...\n")
                await asyncio.sleep(2)
        
        # Final statistics
        logger.info("\n" + "=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)
        
        stats = orchestrator.get_stats()
        logger.info(f"\nOrchestrator Stats:")
        logger.info(f"  Total Queries:       {stats['query_count']}")
        logger.info(f"  Total Time:          {stats['total_processing_time_ms']:.1f} ms")
        logger.info(f"  Avg Time/Query:      {stats['avg_processing_time_ms']:.1f} ms")
        
        llm_stats = stats['llm_stats']
        logger.info(f"\nLLM Stats:")
        logger.info(f"  Total Requests:      {llm_stats['request_count']}")
        logger.info(f"  Total Tokens:        {llm_stats['total_tokens']}")
        logger.info(f"  Avg Tokens/Request:  {llm_stats['avg_tokens_per_request']:.1f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Full pipeline demonstration complete!")
        logger.info("=" * 80)


def display_results(result: dict, test_num: int):
    """Display formatted results."""
    
    logger.info("\n" + "-" * 80)
    logger.info("RESULTS")
    logger.info("-" * 80)
    
    # Context
    logger.info("\n[ONTOLOGICAL CONTEXT]")
    ctx = result['context']
    logger.info(f"  Being:      {ctx['being_aspect']}")
    logger.info(f"  Becoming:   {ctx['becoming_aspect']}")
    logger.info(f"  Suchness:   {ctx['suchness_aspect']}")
    logger.info(f"  Complexity: {ctx['complexity']}")
    logger.info(f"  Domain:     {ctx['domain']}")
    logger.info(f"  Stakes:     {ctx['ethical_stakes']}")
    
    # Experts consulted
    logger.info("\n[EXPERTS CONSULTED]")
    logger.info(f"  Selected: {', '.join(result['experts_consulted'])}")
    
    # Expert results summary
    logger.info("\n[EXPERT RESULTS]")
    for name, data in result['expert_results'].items():
        logger.info(f"  {name.upper()}:")
        logger.info(f"    Consciousness: {data['consciousness']:.3f}")
        logger.info(f"    Coherentia:    {data['coherentia']:.3f}")
        logger.info(f"    Ethical Δ:     {data['ethical_delta']:+.3f}")
    
    # Synthesis
    logger.info("\n[SYNTHESIS]")
    synth = result['synthesis']
    logger.info(f"  Consciousness: {synth['consciousness']:.3f}")
    logger.info(f"  Coherentia:    {synth['coherentia']:.3f}")
    logger.info(f"  Ethical Δ:     {synth['coherentia_delta']:+.3f}")
    logger.info(f"  Ethical:       {synth['ethical_status']}")
    
    # Final response
    logger.info("\n[FINAL RESPONSE]")
    response = result['response']
    if len(response) > 300:
        logger.info(f"  {response[:300]}...")
        logger.info(f"  [Response truncated - {len(response)} chars total]")
    else:
        logger.info(f"  {response}")
    
    # Meta-reflection
    logger.info("\n[META-COGNITIVE REFLECTION]")
    reflection_lines = result['meta_reflection'].split('\n')
    for line in reflection_lines[:10]:  # First 10 lines
        if line.strip():
            logger.info(f"  {line}")
    
    # Ethical evaluation
    logger.info("\n[ETHICAL EVALUATION]")
    eval_lines = result['ethical_evaluation'].split('\n')
    for line in eval_lines[:5]:  # First 5 lines
        if line.strip():
            logger.info(f"  {line}")
    
    # Performance
    logger.info("\n[PERFORMANCE]")
    logger.info(f"  Processing Time: {result['processing_time_ms']:.1f} ms")
    logger.info(f"  Timestamp:       {result['timestamp']}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO"
    )
    
    # Run async main
    asyncio.run(main())
