"""
Demonstration of all 6 LLM-integrated experts.

Shows how each expert processes the same query with different perspectives
based on their Lumen specialization and temperature settings.

Usage:
    python examples/all_experts_demo.py
"""

import asyncio
from loguru import logger

from singularis.llm import LMStudioClient, LMStudioConfig, ExpertLLMInterface
from singularis.tier2_experts import (
    ReasoningExpertLLM,
    CreativeExpertLLM,
    PhilosophicalExpertLLM,
    TechnicalExpertLLM,
    MemoryExpertLLM,
    SynthesisExpertLLM,
)
from singularis.core.types import OntologicalContext


async def main():
    """Run all 6 experts on the same query."""
    
    logger.info("=" * 70)
    logger.info("SINGULARIS - All 6 Experts Demonstration")
    logger.info("=" * 70)
    
    # Configure LM Studio
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="huihui-moe-60b-a38",
        temperature=0.7,
        max_tokens=1024,
    )
    
    # Test query
    query = "How can AI systems achieve genuine understanding?"
    
    logger.info(f"\nQuery: {query}\n")
    
    # Create ontological context
    context = OntologicalContext(
        being_aspect="nature of understanding and intelligence",
        becoming_aspect="development of AI capabilities",
        suchness_aspect="direct recognition of comprehension",
        complexity="high",
        domain="philosophy",
        ethical_stakes="high",
    )
    
    async with LMStudioClient(config) as client:
        llm_interface = ExpertLLMInterface(client)
        
        # Initialize all 6 experts
        experts = {
            "Reasoning": ReasoningExpertLLM(llm_interface, config.model_name),
            "Creative": CreativeExpertLLM(llm_interface, config.model_name),
            "Philosophical": PhilosophicalExpertLLM(llm_interface, config.model_name),
            "Technical": TechnicalExpertLLM(llm_interface, config.model_name),
            "Memory": MemoryExpertLLM(llm_interface, config.model_name),
            "Synthesis": SynthesisExpertLLM(llm_interface, config.model_name),
        }
        
        logger.info("=" * 70)
        logger.info("Processing query with all 6 experts...")
        logger.info("=" * 70 + "\n")
        
        # Process with each expert
        results = {}
        for name, expert in experts.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"EXPERT: {name}")
            logger.info(f"Domain: {expert.domain}")
            logger.info(f"Primary Lumen: {expert.lumen_primary.value}")
            logger.info(f"{'='*70}")
            
            result = await expert.process(
                query=query,
                context=context,
                workspace_coherentia=0.5,
            )
            
            results[name] = result
            
            # Display results
            logger.info(f"\nClaim (first 200 chars):")
            logger.info(result.claim[:200] + "..." if len(result.claim) > 200 else result.claim)
            
            logger.info(f"\nMetrics:")
            logger.info(f"  Confidence:      {result.confidence:.3f}")
            logger.info(f"  Consciousness:   {result.consciousness_trace.overall_consciousness:.3f}")
            logger.info(f"  Coherentia:      {result.coherentia.total:.3f}")
            logger.info(f"  Ethical Delta:   {result.coherentia_delta:+.3f}")
            logger.info(f"  Processing Time: {result.processing_time_ms:.1f} ms")
            
            logger.info(f"\nThree Lumina:")
            logger.info(f"  Ontical:       {result.coherentia.ontical:.3f}")
            logger.info(f"  Structural:    {result.coherentia.structural:.3f}")
            logger.info(f"  Participatory: {result.coherentia.participatory:.3f}")
        
        # Summary comparison
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY COMPARISON")
        logger.info("=" * 70)
        
        logger.info(f"\n{'Expert':<15} {'Confidence':<12} {'Consciousness':<15} {'Coherentia':<12} {'Ethical':<10}")
        logger.info("-" * 70)
        
        for name, result in results.items():
            ethical = "ETHICAL" if result.ethical_status else "NEUTRAL" if result.ethical_status is None else "UNETHICAL"
            logger.info(
                f"{name:<15} "
                f"{result.confidence:<12.3f} "
                f"{result.consciousness_trace.overall_consciousness:<15.3f} "
                f"{result.coherentia.total:<12.3f} "
                f"{ethical:<10}"
            )
        
        # Find highest coherence
        best_expert = max(results.items(), key=lambda x: x[1].coherentia.total)
        logger.info(f"\nHighest Coherentia: {best_expert[0]} ({best_expert[1].coherentia.total:.3f})")
        
        # LLM statistics
        stats = client.get_stats()
        logger.info("\n" + "=" * 70)
        logger.info("LLM STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total Requests:  {stats['request_count']}")
        logger.info(f"Total Tokens:    {stats['total_tokens']}")
        logger.info(f"Avg per Request: {stats['avg_tokens_per_request']:.1f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Demonstration complete!")
        logger.info("=" * 70)


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
