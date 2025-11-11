"""
Test Learning System

Quick test to verify text processing and learning components work.

Usage:
    python examples/test_learning.py
"""

import asyncio
from pathlib import Path
from loguru import logger

from singularis.learning import TextProcessor, CurriculumLoader, LearningProgress


def test_text_processor():
    """Test text chunking."""
    logger.info("=" * 70)
    logger.info("TEST 1: Text Processor")
    logger.info("=" * 70)
    
    processor = TextProcessor(max_chunk_size=1000, overlap=100)
    
    # Sample text
    sample_text = """
    Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language.
    
    Ancient Greek philosophers like Socrates, Plato, and Aristotle laid the foundations for Western philosophy.
    
    Socrates emphasized the importance of self-knowledge and critical thinking. His method of questioning, known as the Socratic method, remains influential today.
    
    Plato, a student of Socrates, founded the Academy in Athens and wrote extensively on topics including justice, beauty, and the nature of reality.
    
    Aristotle, Plato's student, made significant contributions to logic, metaphysics, ethics, and natural sciences.
    
    Medieval philosophy was dominated by religious thought, with figures like Augustine and Aquinas attempting to reconcile faith and reason.
    
    The Enlightenment brought a renewed focus on reason and empiricism, with philosophers like Descartes, Locke, and Kant shaping modern thought.
    
    Contemporary philosophy addresses issues ranging from consciousness and artificial intelligence to social justice and environmental ethics.
    """ * 3  # Repeat to make it longer
    
    chunks = processor.chunk_text(
        text=sample_text,
        source="test_philosophy.txt",
        metadata={"test": True}
    )
    
    logger.info(f"✓ Created {len(chunks)} chunks")
    logger.info(f"✓ Chunk sizes: {[len(c.text) for c in chunks]}")
    logger.info(f"✓ First chunk preview: {chunks[0].text[:100]}...")
    
    return chunks


def test_curriculum_loader():
    """Test curriculum loading."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Curriculum Loader")
    logger.info("=" * 70)
    
    project_root = Path(__file__).parent.parent
    curriculum_dir = project_root / "university_curriculum"
    
    if not curriculum_dir.exists():
        logger.warning("Curriculum directory not found")
        return
    
    curriculum = CurriculumLoader(curriculum_dir)
    
    domains = curriculum.get_domains()
    logger.info(f"✓ Found {len(domains)} domains")
    logger.info(f"✓ Sample domains: {domains[:5]}")
    
    # Test loading files from first domain
    if domains:
        first_domain = domains[0]
        files = curriculum.get_domain_files(first_domain)
        logger.info(f"✓ Domain '{first_domain}' has {len(files)} files")
        if files:
            logger.info(f"✓ Sample file: {files[0].name}")
    
    return curriculum


def test_learning_progress():
    """Test progress tracking."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Learning Progress")
    logger.info("=" * 70)
    
    import tempfile
    temp_file = Path(tempfile.mktemp(suffix=".json"))
    
    progress = LearningProgress(temp_file)
    
    # Mark some chunks as processed
    for i in range(5):
        progress.mark_processed(
            source=f"test_text_{i}.txt",
            chunk_index=i,
            coherentia=0.65 + (i * 0.05),
            processing_time_ms=1000.0 + (i * 100),
            ethical=i % 2 == 0,
        )
    
    progress.save()
    
    stats = progress.get_stats()
    logger.info(f"✓ Chunks processed: {stats['chunks_processed']}")
    logger.info(f"✓ Avg coherentia: {stats['avg_coherentia']:.3f}")
    logger.info(f"✓ Ethical rate: {stats['ethical_rate']:.1%}")
    
    # Clean up
    temp_file.unlink()
    
    return progress


def test_philosophy_texts():
    """Test loading philosophy texts."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Philosophy Texts")
    logger.info("=" * 70)
    
    project_root = Path(__file__).parent.parent
    philosophy_dir = project_root / "philosophy_texts"
    
    if not philosophy_dir.exists():
        logger.warning("Philosophy texts directory not found")
        return
    
    processor = TextProcessor(max_chunk_size=3000, overlap=200)
    
    files = list(philosophy_dir.glob("*.txt"))
    logger.info(f"✓ Found {len(files)} philosophy texts")
    
    if files:
        # Test first file
        first_file = files[0]
        logger.info(f"✓ Testing: {first_file.name}")
        
        text = processor.load_text_file(first_file)
        logger.info(f"✓ Loaded {len(text)} characters")
        
        chunks = processor.chunk_text(
            text=text,
            source=first_file.name,
        )
        logger.info(f"✓ Created {len(chunks)} chunks")
        logger.info(f"✓ First chunk: {chunks[0].text[:100]}...")


async def test_with_llm():
    """Test with actual LLM (if available)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: LLM Integration (Optional)")
    logger.info("=" * 70)
    
    try:
        from singularis.llm import LMStudioClient, LMStudioConfig
        from singularis.tier1_orchestrator import MetaOrchestratorLLM
        
        config = LMStudioConfig(
            base_url="http://localhost:1234/v1",
            model_name="huihui-moe-60b-a38",
        )
        
        # Test connection
        async with LMStudioClient(config) as client:
            logger.info("✓ LM Studio client created")
            
            orchestrator = MetaOrchestratorLLM(
                llm_client=client,
                consciousness_threshold=0.65,
            )
            logger.info("✓ MetaOrchestrator initialized")
            
            # Simple test query
            result = await orchestrator.process(
                "What is the essence of philosophical inquiry?"
            )
            
            logger.info(f"✓ Query processed")
            logger.info(f"✓ Coherentia: {result['synthesis']['coherentia']:.3f}")
            logger.info(f"✓ Experts: {result['experts_consulted']}")
            
    except Exception as e:
        logger.warning(f"LLM test skipped: {e}")
        logger.info("(This is optional - requires LM Studio running)")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("SINGULARIS LEARNING SYSTEM - TEST SUITE")
    logger.info("=" * 70)
    
    try:
        # Test 1: Text Processor
        chunks = test_text_processor()
        
        # Test 2: Curriculum Loader
        curriculum = test_curriculum_loader()
        
        # Test 3: Learning Progress
        progress = test_learning_progress()
        
        # Test 4: Philosophy Texts
        test_philosophy_texts()
        
        # Test 5: LLM Integration (optional)
        asyncio.run(test_with_llm())
        
        logger.info("\n" + "=" * 70)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 70)
        logger.info("\nLearning system is ready!")
        logger.info("\nNext steps:")
        logger.info("  1. Start LM Studio with Huihui MoE 60B")
        logger.info("  2. Run: python examples/batch_learn.py --source philosophy --limit 10")
        logger.info("  3. Monitor progress in learning_progress.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO"
    )
    
    main()
