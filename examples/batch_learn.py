"""
Batch Learning with Progress Tracking and Resume Capability

Processes texts in batches with:
- Resume capability (skip already processed)
- Detailed progress tracking
- Configurable batch sizes
- Error recovery

Usage:
    # Process first 10 chunks of philosophy texts
    python examples/batch_learn.py --source philosophy --limit 10
    
    # Process specific curriculum domains
    python examples/batch_learn.py --source curriculum --domains ethics_&_moral_philosophy epistemology_&_metaphysics
    
    # Resume previous session
    python examples/batch_learn.py --resume
"""

import asyncio
import argparse
from pathlib import Path
from loguru import logger
import time
import json
from datetime import datetime

from singularis.llm import LMStudioClient, LMStudioConfig
from singularis.tier1_orchestrator import MetaOrchestratorLLM
from singularis.learning import TextProcessor, CurriculumLoader, LearningProgress
from singularis.tier3_neurons import NeuronSwarm, Lumen


class BatchLearner:
    """
    Batch learning system with resume capability.
    
    Philosophy:
    Learning is incremental. Each batch increases understanding.
    Progress is preserved across sessions.
    """
    
    def __init__(
        self,
        orchestrator: MetaOrchestratorLLM,
        neuron_swarm: NeuronSwarm,
        progress_file: Path,
    ):
        """Initialize batch learner."""
        self.orchestrator = orchestrator
        self.neuron_swarm = neuron_swarm
        self.progress = LearningProgress(progress_file)
        self.processed_chunks = set()
        
        # Load processed chunks from progress
        self._load_processed_chunks()
    
    def _load_processed_chunks(self):
        """Load set of already processed chunks."""
        # Create unique IDs for processed chunks
        if "processed_chunks" in self.progress.progress:
            self.processed_chunks = set(self.progress.progress["processed_chunks"])
        else:
            self.progress.progress["processed_chunks"] = []
    
    def _chunk_id(self, source: str, chunk_index: int) -> str:
        """Create unique ID for a chunk."""
        return f"{source}:{chunk_index}"
    
    def is_processed(self, source: str, chunk_index: int) -> bool:
        """Check if chunk was already processed."""
        return self._chunk_id(source, chunk_index) in self.processed_chunks
    
    async def process_chunk(self, chunk, domain: str = "general"):
        """Process a single chunk."""
        # Check if already processed
        if self.is_processed(chunk.source, chunk.chunk_index):
            logger.debug(f"Skipping already processed: {chunk.source} [{chunk.chunk_index}]")
            return None
        
        # Create learning query
        query = f"""Analyze this text from {domain}:

Source: {chunk.source}
Section: {chunk.chunk_index + 1}/{chunk.total_chunks}

{chunk.text[:2500]}

Extract and integrate key concepts, arguments, and philosophical insights."""
        
        logger.info(
            f"Processing: {chunk.source} [{chunk.chunk_index + 1}/{chunk.total_chunks}]"
        )
        
        try:
            # Process through orchestrator
            result = await self.orchestrator.process(query)
            
            # Determine Lumen focus
            if 'ethics' in domain.lower() or 'moral' in domain.lower():
                lumen_focus = Lumen.PARTICIPATUM
            elif 'logic' in domain.lower() or 'math' in domain.lower():
                lumen_focus = Lumen.STRUCTURALE
            else:
                lumen_focus = Lumen.ONTICUM
            
            # Learn in neuron swarm
            concepts = result['response'][:500]
            neuron_result = self.neuron_swarm.process_pattern(
                pattern=concepts,
                lumen_focus=lumen_focus,
                iterations=2,
            )
            
            # Mark as processed
            chunk_id = self._chunk_id(chunk.source, chunk.chunk_index)
            self.processed_chunks.add(chunk_id)
            self.progress.progress["processed_chunks"].append(chunk_id)
            
            # Track progress
            self.progress.mark_processed(
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                coherentia=result['synthesis']['coherentia'],
                processing_time_ms=result['processing_time_ms'],
                ethical=result['synthesis']['ethical_status'],
            )
            
            logger.success(
                f"✓ {chunk.source} [{chunk.chunk_index + 1}/{chunk.total_chunks}] | "
                f"C={result['synthesis']['coherentia']:.3f} | "
                f"Neurons={neuron_result['active_neurons']}/18"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"✗ Failed: {chunk.source} [{chunk.chunk_index + 1}]: {e}"
            )
            return None
    
    async def process_batch(
        self,
        chunks,
        domain: str = "general",
        batch_size: int = 10,
        save_interval: int = 5,
    ):
        """
        Process a batch of chunks.
        
        Args:
            chunks: List of chunks to process
            domain: Domain name
            batch_size: Max chunks per batch
            save_interval: Save progress every N chunks
        """
        processed = 0
        skipped = 0
        failed = 0
        
        for i, chunk in enumerate(chunks):
            if processed >= batch_size:
                logger.info(f"Batch limit reached ({batch_size})")
                break
            
            if self.is_processed(chunk.source, chunk.chunk_index):
                skipped += 1
                continue
            
            result = await self.process_chunk(chunk, domain)
            
            if result:
                processed += 1
            else:
                failed += 1
            
            # Save progress periodically
            if (processed + failed) % save_interval == 0:
                self.progress.save()
                logger.info(f"Progress saved ({processed} processed, {skipped} skipped, {failed} failed)")
            
            # Brief pause
            await asyncio.sleep(0.5)
        
        # Final save
        self.progress.save()
        
        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        }


async def main():
    """Main batch learning function."""
    parser = argparse.ArgumentParser(description="Batch learning from texts")
    parser.add_argument(
        '--source',
        choices=['philosophy', 'curriculum'],
        default='philosophy',
        help='Source to learn from'
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        help='Specific curriculum domains'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Maximum chunks to process in this batch'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=3000,
        help='Maximum characters per chunk'
    )
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    philosophy_dir = project_root / "philosophy_texts"
    curriculum_dir = project_root / "university_curriculum"
    progress_file = project_root / "learning_progress.json"
    
    logger.info("=" * 80)
    logger.info("SINGULARIS BATCH LEARNING")
    logger.info("=" * 80)
    logger.info(f"Source: {args.source}")
    logger.info(f"Batch limit: {args.limit} chunks")
    logger.info(f"Chunk size: {args.chunk_size} chars")
    if args.resume:
        logger.info("Resume mode: ON")
    logger.info("=" * 80)
    
    # Initialize LM Studio
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="huihui-moe-60b-a38",
        temperature=0.7,
        max_tokens=1024,
    )
    
    async with LMStudioClient(config) as client:
        # Initialize systems
        orchestrator = MetaOrchestratorLLM(
            llm_client=client,
            consciousness_threshold=0.65,
            coherentia_threshold=0.60,
            ethical_threshold=0.02,
        )
        
        neuron_swarm = NeuronSwarm(
            neurons_per_layer=6,
            learning_rate=0.05,
            activation_threshold=0.5,
        )
        
        learner = BatchLearner(orchestrator, neuron_swarm, progress_file)
        
        # Show current progress
        stats = learner.progress.get_stats()
        logger.info(f"\nCurrent progress:")
        logger.info(f"  Texts: {stats['texts_processed']}")
        logger.info(f"  Chunks: {stats['chunks_processed']}")
        logger.info(f"  Avg coherentia: {stats['avg_coherentia']:.3f}")
        logger.info(f"  Ethical rate: {stats['ethical_rate']:.1%}\n")
        
        start_time = time.time()
        
        # Process based on source
        processor = TextProcessor(max_chunk_size=args.chunk_size, overlap=200)
        
        if args.source == 'philosophy':
            logger.info("Processing philosophy texts...")
            
            for filename, chunks in processor.process_directory(philosophy_dir):
                logger.info(f"\n📖 {filename} ({len(chunks)} chunks)")
                
                result = await learner.process_batch(
                    chunks=chunks,
                    domain="philosophy",
                    batch_size=args.limit,
                )
                
                logger.info(
                    f"Batch complete: {result['processed']} processed, "
                    f"{result['skipped']} skipped, {result['failed']} failed"
                )
                
                if result['processed'] >= args.limit:
                    break
        
        elif args.source == 'curriculum':
            logger.info("Processing curriculum...")
            
            curriculum = CurriculumLoader(curriculum_dir)
            domains = args.domains if args.domains else curriculum.get_domains()[:5]  # First 5 by default
            
            logger.info(f"Selected domains: {domains}")
            
            total_processed = 0
            
            for domain, filename, chunks in curriculum.iterate_curriculum(processor, domains):
                logger.info(f"\n📚 {domain}/{filename} ({len(chunks)} chunks)")
                
                remaining = args.limit - total_processed
                if remaining <= 0:
                    break
                
                result = await learner.process_batch(
                    chunks=chunks,
                    domain=domain,
                    batch_size=remaining,
                )
                
                total_processed += result['processed']
                
                logger.info(
                    f"Batch complete: {result['processed']} processed, "
                    f"{result['skipped']} skipped, {result['failed']} failed"
                )
                
                if total_processed >= args.limit:
                    break
        
        # Final statistics
        elapsed = time.time() - start_time
        final_stats = learner.progress.get_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("BATCH COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Time elapsed: {elapsed / 60:.1f} minutes")
        logger.info(f"Total chunks processed: {final_stats['chunks_processed']}")
        logger.info(f"Average coherentia: {final_stats['avg_coherentia']:.3f}")
        logger.info(f"Ethical rate: {final_stats['ethical_rate']:.1%}")
        
        # Neuron statistics
        swarm_stats = neuron_swarm.get_statistics()
        logger.info(f"\nNeuron swarm:")
        logger.info(f"  Connections: {swarm_stats['total_connections']}")
        logger.info(f"  Patterns learned: {swarm_stats['patterns_learned']}")
        logger.info(f"  Avg connections/neuron: {swarm_stats['avg_connections_per_neuron']:.1f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Progress saved. Run again to continue learning.")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO"
    )
    
    # Run
    asyncio.run(main())
