"""
Quickstart: Singularis with LM Studio Integration

This example shows how to use Singularis with Huihui MoE 60B via LM Studio.

Prerequisites:
1. LM Studio running on localhost:1234
2. Huihui MoE 60B model loaded
3. Server started in LM Studio

Usage:
    python examples/quickstart_llm.py
"""

import asyncio
from loguru import logger

from singularis.llm import LMStudioClient, LMStudioConfig, ExpertLLMInterface
from singularis.tier2_experts.reasoning_expert_llm import ReasoningExpertLLM
from singularis.core.types import OntologicalContext


async def main():
    """Run quickstart example."""
    
    logger.info("=" * 60)
    logger.info("SINGULARIS - LM Studio Integration Quickstart")
    logger.info("=" * 60)
    
    # Step 1: Configure LM Studio connection
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="huihui-moe-60b-a38",  # Adjust to your model name in LM Studio
        temperature=0.7,
        max_tokens=2048,
    )
    
    logger.info(f"Connecting to LM Studio: {config.base_url}")
    logger.info(f"Model: {config.model_name}")
    
    # Step 2: Create LM Studio client
    async with LMStudioClient(config) as client:
        
        # Step 3: Create expert LLM interface
        llm_interface = ExpertLLMInterface(client)
        
        # Step 4: Initialize Reasoning Expert with LLM
        reasoning_expert = ReasoningExpertLLM(
            llm_interface=llm_interface,
            model_id=config.model_name
        )
        
        logger.info(f"Expert initialized: {reasoning_expert.name}")
        logger.info(f"Domain: {reasoning_expert.domain}")
        logger.info(f"Primary Lumen: {reasoning_expert.lumen_primary.value}")
        
        # Step 5: Create test query
        query = "What is the relationship between consciousness and coherence in Spinoza's philosophy?"
        
        logger.info("\n" + "=" * 60)
        logger.info("QUERY:")
        logger.info(query)
        logger.info("=" * 60 + "\n")
        
        # Step 6: Create ontological context
        context = OntologicalContext(
            being_aspect="consciousness and coherence as unified",
            becoming_aspect="understanding through participation",
            suchness_aspect="direct recognition of necessity",
            complexity="high",
            domain="philosophy",
            ethical_stakes="medium",
        )
        
        # Step 7: Process query
        logger.info("Processing query with Reasoning Expert...")
        
        result = await reasoning_expert.process(
            query=query,
            context=context,
            workspace_coherentia=0.5,
        )
        
        # Step 8: Display results
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS:")
        logger.info("=" * 60)
        
        # Use logger instead of print to avoid encoding issues
        logger.info(f"\n{'Expert:':<20} {result.expert_name}")
        logger.info(f"{'Domain:':<20} {result.domain}")
        logger.info(f"{'Primary Lumen:':<20} {result.lumen_primary.value}")
        logger.info(f"\n{'CLAIM:':<20}")
        logger.info(result.claim)
        logger.info(f"\n{'RATIONALE:':<20}")
        logger.info(result.rationale)
        logger.info(f"\n{'Confidence:':<20} {result.confidence:.3f}")
        
        logger.info(f"\n{'CONSCIOUSNESS METRICS:'}")
        logger.info(f"{'  Overall:':<20} {result.consciousness_trace.overall_consciousness:.3f}")
        logger.info(f"{'  IIT Phi:':<20} {result.consciousness_trace.iit_phi:.3f}")
        logger.info(f"{'  GWT Salience:':<20} {result.consciousness_trace.gwt_salience:.3f}")
        logger.info(f"{'  HOT Depth:':<20} {result.consciousness_trace.hot_reflection_depth:.3f}")
        
        logger.info(f"\n{'COHERENTIA (Three Lumina):'}")
        logger.info(f"{'  Ontical (lo):':<20} {result.coherentia.ontical:.3f}")
        logger.info(f"{'  Structural (ls):':<20} {result.coherentia.structural:.3f}")
        logger.info(f"{'  Participatory (lp):':<20} {result.coherentia.participatory:.3f}")
        logger.info(f"{'  Total (C):':<20} {result.coherentia.total:.3f}")
        
        logger.info(f"\n{'ETHICAL VALIDATION:'}")
        logger.info(f"{'  Coherentia Delta:':<20} {result.coherentia_delta:+.3f}")
        logger.info(f"{'  Ethical Status:':<20} {result.ethical_status}")
        logger.info(f"{'  Reasoning:':<20} {result.ethical_reasoning}")
        
        logger.info(f"\n{'PERFORMANCE:'}")
        logger.info(f"{'  Processing Time:':<20} {result.processing_time_ms:.1f} ms")
        
        # Step 9: Show client stats
        stats = client.get_stats()
        logger.info(f"\n{'LLM STATISTICS:'}")
        logger.info(f"{'  Total Requests:':<20} {stats['request_count']}")
        logger.info(f"{'  Total Tokens:':<20} {stats['total_tokens']}")
        logger.info(f"{'  Avg Tokens/Req:':<20} {stats['avg_tokens_per_request']:.1f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Quickstart complete!")
        logger.info("=" * 60)


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
