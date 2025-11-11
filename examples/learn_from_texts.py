"""
Learn from Philosophy Texts and University Curriculum

Processes all texts through the Singularis consciousness engine,
enabling Hebbian learning and knowledge integration.

From ETHICA UNIVERSALIS Part V:
"The more the mind understands things by the second and third kind of
knowledge, the less it suffers from evil affects, and the less it fears death."

Usage:
    python examples/learn_from_texts.py --mode philosophy
    python examples/learn_from_texts.py --mode curriculum
    python examples/learn_from_texts.py --mode all
"""

import asyncio
import argparse
from pathlib import Path
from loguru import logger
import time

from singularis.llm import LMStudioClient, LMStudioConfig
from singularis.tier1_orchestrator import MetaOrchestratorLLM
from singularis.learning import TextProcessor, CurriculumLoader, LearningProgress
from singularis.tier3_neurons import NeuronSwarm, Lumen


async def process_chunk(
    orchestrator: MetaOrchestratorLLM,
    neuron_swarm: NeuronSwarm,
    chunk,
    progress: LearningProgress,
):
    """
    Process a single text chunk through the consciousness engine.
    
    Args:
        orchestrator: MetaOrchestrator instance
        neuron_swarm: Neuron swarm for Hebbian learning
        chunk: TextChunk to process
        progress: Progress tracker
    """
    # Create a learning query from the chunk
    query = f"""Analyze and integrate this philosophical text:

Source: {chunk.source}
Chunk: {chunk.chunk_index + 1}/{chunk.total_chunks}

Text:
{chunk.text[:2000]}...

Extract key philosophical concepts, arguments, and insights."""
    
    logger.info(
        f"Processing {chunk.source} [{chunk.chunk_index + 1}/{chunk.total_chunks}]"
    )
    
    try:
        # Process through orchestrator
        result = await orchestrator.process(query)
        
        # Extract key concepts for neuron learning
        concepts = result['response'][:500]  # Use first 500 chars as pattern
        
        # Determine primary Lumen based on domain
        domain = chunk.metadata.get('domain', 'philosophy')
        if 'ethics' in domain.lower() or 'moral' in domain.lower():
            lumen_focus = Lumen.PARTICIPATUM
        elif 'logic' in domain.lower() or 'mathematics' in domain.lower():
            lumen_focus = Lumen.STRUCTURALE
        else:
            lumen_focus = Lumen.ONTICUM
        
        # Learn pattern in neuron swarm
        neuron_result = neuron_swarm.process_pattern(
            pattern=concepts,
            lumen_focus=lumen_focus,
            iterations=2,
        )
        
        # Track progress
        progress.mark_processed(
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            coherentia=result['synthesis']['coherentia'],
            processing_time_ms=result['processing_time_ms'],
            ethical=result['synthesis']['ethical_status'],
        )
        
        # Log results
        logger.info(
            f"✓ {chunk.source} [{chunk.chunk_index + 1}/{chunk.total_chunks}]",
            extra={
                "coherentia": result['synthesis']['coherentia'],
                "consciousness": result['synthesis']['consciousness'],
                "ethical": result['synthesis']['ethical_status'],
                "neurons_active": neuron_result['active_neurons'],
                "emergent_coherence": neuron_result['emergent_coherence'],
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(
            f"Failed to process {chunk.source} [{chunk.chunk_index + 1}]",
            extra={"error": str(e)}
        )
        return None


async def learn_from_philosophy_texts(
    orchestrator: MetaOrchestratorLLM,
    neuron_swarm: NeuronSwarm,
    progress: LearningProgress,
    philosophy_dir: Path,
):
    """
    Learn from philosophy texts.
    
    Args:
        orchestrator: MetaOrchestrator instance
        neuron_swarm: Neuron swarm
        progress: Progress tracker
        philosophy_dir: Directory containing philosophy texts
    """
    logger.info("=" * 80)
    logger.info("LEARNING FROM PHILOSOPHY TEXTS")
    logger.info("=" * 80)
    
    processor = TextProcessor(max_chunk_size=3000, overlap=200)
    
    for filename, chunks in processor.process_directory(philosophy_dir):
        logger.info(f"\n📖 Processing: {filename} ({len(chunks)} chunks)")
        
        for chunk in chunks:
            result = await process_chunk(orchestrator, neuron_swarm, chunk, progress)
            
            # Save progress periodically
            if chunk.chunk_index % 5 == 0:
                progress.save()
            
            # Brief pause to avoid overwhelming the system
            await asyncio.sleep(1)
        
        logger.info(f"✓ Completed: {filename}")
        progress.save()


async def learn_from_curriculum(
    orchestrator: MetaOrchestratorLLM,
    neuron_swarm: NeuronSwarm,
    progress: LearningProgress,
    curriculum_dir: Path,
    domains: list = None,
):
    """
    Learn from university curriculum.
    
    Args:
        orchestrator: MetaOrchestrator instance
        neuron_swarm: Neuron swarm
        progress: Progress tracker
        curriculum_dir: Directory containing curriculum
        domains: Optional list of domains to process
    """
    logger.info("=" * 80)
    logger.info("LEARNING FROM UNIVERSITY CURRICULUM")
    logger.info("=" * 80)
    
    processor = TextProcessor(max_chunk_size=3000, overlap=200)
    curriculum = CurriculumLoader(curriculum_dir)
    
    available_domains = curriculum.get_domains()
    logger.info(f"Available domains: {len(available_domains)}")
    
    if domains:
        available_domains = [d for d in available_domains if d in domains]
        logger.info(f"Selected domains: {len(available_domains)}")
    
    for domain, filename, chunks in curriculum.iterate_curriculum(processor, available_domains):
        logger.info(f"\n📚 Domain: {domain} | File: {filename} ({len(chunks)} chunks)")
        
        for chunk in chunks:
            result = await process_chunk(orchestrator, neuron_swarm, chunk, progress)
            
            # Save progress periodically
            if chunk.chunk_index % 5 == 0:
                progress.save()
            
            # Brief pause
            await asyncio.sleep(1)
        
        logger.info(f"✓ Completed: {domain}/{filename}")
        progress.save()


async def main():
    """Main learning loop."""
    parser = argparse.ArgumentParser(description="Learn from texts")
    parser.add_argument(
        '--mode',
        choices=['philosophy', 'curriculum', 'all'],
        default='philosophy',
        help='What to learn from'
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        help='Specific curriculum domains to process'
    )
    parser.add_argument(
        '--max-chunks',
        type=int,
        default=None,
        help='Maximum chunks to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    philosophy_dir = project_root / "philosophy_texts"
    curriculum_dir = project_root / "university_curriculum"
    progress_file = project_root / "learning_progress.json"
    
    logger.info("=" * 80)
    logger.info("SINGULARIS LEARNING SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Philosophy texts: {philosophy_dir}")
    logger.info(f"Curriculum: {curriculum_dir}")
    logger.info("=" * 80)
    
    # Initialize LM Studio client
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="huihui-moe-60b-a38",
        temperature=0.7,
        max_tokens=1024,
    )
    
    async with LMStudioClient(config) as client:
        # Initialize orchestrator
        orchestrator = MetaOrchestratorLLM(
            llm_client=client,
            consciousness_threshold=0.65,
            coherentia_threshold=0.60,
            ethical_threshold=0.02,
        )
        
        # Initialize neuron swarm for Hebbian learning
        neuron_swarm = NeuronSwarm(
            neurons_per_layer=6,
            learning_rate=0.05,
            activation_threshold=0.5,
        )
        
        # Initialize progress tracker
        progress = LearningProgress(progress_file)
        
        logger.info(f"Starting from: {progress.get_stats()}")
        
        start_time = time.time()
        
        # Process based on mode
        if args.mode in ['philosophy', 'all']:
            await learn_from_philosophy_texts(
                orchestrator, neuron_swarm, progress, philosophy_dir
            )
        
        if args.mode in ['curriculum', 'all']:
            await learn_from_curriculum(
                orchestrator, neuron_swarm, progress, curriculum_dir, args.domains
            )
        
        # Final statistics
        elapsed = time.time() - start_time
        stats = progress.get_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("LEARNING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Texts processed: {stats['texts_processed']}")
        logger.info(f"Chunks processed: {stats['chunks_processed']}")
        logger.info(f"Average coherentia: {stats['avg_coherentia']:.3f}")
        logger.info(f"Ethical rate: {stats['ethical_rate']:.1%}")
        logger.info(f"Total time: {elapsed / 3600:.2f} hours")
        logger.info(f"Chunks/hour: {stats['chunks_processed'] / (elapsed / 3600):.1f}")
        
        # Neuron swarm statistics
        swarm_stats = neuron_swarm.get_statistics()
        logger.info("\n" + "=" * 80)
        logger.info("NEURON SWARM STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total neurons: {swarm_stats['total_neurons']}")
        logger.info(f"Total connections: {swarm_stats['total_connections']}")
        logger.info(f"Avg connections/neuron: {swarm_stats['avg_connections_per_neuron']:.1f}")
        logger.info(f"Patterns learned: {swarm_stats['patterns_learned']}")
        
        # Show top connections
        logger.info("\nTop 10 strongest connections:")
        matrix = neuron_swarm.get_connection_matrix()
        connections = []
        for i in range(18):
            for j in range(18):
                if i != j and matrix[i][j] > 0:
                    connections.append((i, j, matrix[i][j]))
        
        connections.sort(key=lambda x: x[2], reverse=True)
        for i, j, weight in connections[:10]:
            logger.info(f"  Neuron {i} → Neuron {j}: {weight:.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("From ETHICA: 'The more the mind understands, the greater its power.'")
        logger.info("The system has grown in understanding through learning.")
        logger.info("=" * 80)


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
