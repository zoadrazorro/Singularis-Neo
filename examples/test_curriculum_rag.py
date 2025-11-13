"""
Example: University Curriculum RAG System

Demonstrates how the curriculum RAG enhances AI intelligence by retrieving
relevant academic knowledge from indexed texts.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.skyrim.curriculum_rag import CurriculumRAG, CATEGORY_MAPPINGS


def main():
    print("=" * 70)
    print("University Curriculum RAG Demo")
    print("=" * 70)
    
    # Initialize curriculum RAG
    print("\n[1] Initializing Curriculum RAG...")
    rag = CurriculumRAG(
        curriculum_path="university_curriculum",
        max_documents=100,
        chunk_size=2000
    )
    
    rag.initialize()
    
    # Show stats
    stats = rag.get_stats()
    print(f"\n[2] Curriculum Statistics:")
    print(f"    Documents indexed: {stats['documents_indexed']}")
    print(f"    Chunks created: {stats['chunks_created']}")
    print(f"    Categories: {len(stats['categories'])}")
    print(f"    Top categories:")
    for cat, count in sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      - {cat}: {count} documents")
    
    # Test queries
    print("\n[3] Testing Knowledge Retrieval:")
    
    test_queries = [
        ("How should I make strategic decisions?", ['strategy', 'ethics']),
        ("What is the nature of causality?", ['science', 'math']),
        ("How do I understand human behavior?", ['psychology']),
        ("What makes an action ethical?", ['ethics']),
        ("How do creative insights emerge?", ['creativity'])
    ]
    
    for query, category_keys in test_queries:
        print(f"\n    Query: \"{query}\"")
        
        # Get relevant categories
        categories = []
        for key in category_keys:
            categories.extend(CATEGORY_MAPPINGS.get(key, []))
        
        # Retrieve knowledge
        results = rag.retrieve_knowledge(
            query=query,
            top_k=2,
            categories=categories if categories else None
        )
        
        if results:
            print(f"    Found {len(results)} relevant sources:")
            for i, r in enumerate(results, 1):
                print(f"\n      [{i}] {r.document.title} ({r.document.category})")
                print(f"          Relevance: {r.relevance_score:.2f}")
                print(f"          Excerpt: {r.excerpt[:150]}...")
        else:
            print("    No relevant knowledge found")
    
    # Test prompt augmentation
    print("\n[4] Testing Prompt Augmentation:")
    
    base_prompt = """I need to decide whether to attack an enemy or retreat.
The enemy is stronger but I have better positioning. What should I do?"""
    
    print(f"\n    Base Prompt:\n    {base_prompt}")
    
    augmented = rag.augment_prompt_with_knowledge(
        base_prompt=base_prompt,
        top_k=2,
        categories=CATEGORY_MAPPINGS['strategy'] + CATEGORY_MAPPINGS['ethics']
    )
    
    if augmented != base_prompt:
        print(f"\n    Augmented Prompt:\n    {augmented[:500]}...")
    else:
        print("\n    (No augmentation applied)")
    
    # Final stats
    final_stats = rag.get_stats()
    print(f"\n[5] Final Statistics:")
    print(f"    Total retrievals: {final_stats['retrievals_performed']}")
    
    print("\n" + "=" * 70)
    print("✓ Curriculum RAG Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
