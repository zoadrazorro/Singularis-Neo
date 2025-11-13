"""
University Curriculum RAG - Enhanced Intelligence through Academic Knowledge

This system indexes and retrieves relevant knowledge from the university curriculum
to augment the AI's decision-making with deep academic understanding across:
- Sciences (natural, physical, advanced)
- Mathematics and Logic
- Philosophy and Ethics
- Literature and Arts
- History and Social Sciences
- Psychology and Economics

The curriculum knowledge enhances:
1. Strategic reasoning (philosophy, logic)
2. Causal understanding (science, mathematics)
3. Ethical decision-making (ethics, moral philosophy)
4. Social dynamics (psychology, sociology)
5. Creative problem-solving (literature, arts)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict

from loguru import logger


@dataclass
class CurriculumDocument:
    """A document from the university curriculum."""
    text_id: str
    category: str
    title: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


@dataclass
class KnowledgeRetrieval:
    """Retrieved knowledge from curriculum."""
    document: CurriculumDocument
    relevance_score: float
    excerpt: str  # Relevant excerpt from document


class CurriculumRAG:
    """
    RAG system for university curriculum knowledge.
    
    Provides academic knowledge retrieval to enhance AI intelligence.
    """
    
    def __init__(
        self,
        curriculum_path: str = "university_curriculum",
        max_documents: int = 200,
        chunk_size: int = 2000,
        use_embeddings: bool = False
    ):
        """
        Initialize curriculum RAG system.
        
        Args:
            curriculum_path: Path to curriculum directory
            max_documents: Maximum documents to index
            chunk_size: Size of text chunks for retrieval
            use_embeddings: Use embeddings for similarity (requires sentence-transformers)
        """
        self.curriculum_path = Path(curriculum_path)
        self.max_documents = max_documents
        self.chunk_size = chunk_size
        self.use_embeddings = use_embeddings
        
        # Document storage
        self.documents: List[CurriculumDocument] = []
        self.document_chunks: List[Tuple[CurriculumDocument, str, int]] = []  # (doc, chunk, chunk_idx)
        
        # Category index for targeted retrieval
        self.category_index: Dict[str, List[int]] = defaultdict(list)
        
        # Keyword index for fast lookup
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        
        # Embedding model (optional)
        self.embedder = None
        
        # Stats
        self.retrieval_count = 0
        self.cache_hits = 0
        
        logger.info(f"[CURRICULUM-RAG] Initializing with path: {curriculum_path}")
    
    def initialize(self):
        """Load and index curriculum documents."""
        logger.info("[CURRICULUM-RAG] Loading university curriculum...")
        
        # Load manifest
        manifest_path = self.curriculum_path / "curriculum_manifest.json"
        if not manifest_path.exists():
            logger.warning(f"[CURRICULUM-RAG] Manifest not found: {manifest_path}")
            return
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        completed = manifest.get('completed', [])
        categories = manifest.get('categories', {})
        
        logger.info(f"[CURRICULUM-RAG] Found {len(completed)} completed texts")
        
        # Index documents by category
        documents_loaded = 0
        for text_id in completed[:self.max_documents]:
            # Find the text file
            text_file = self._find_text_file(text_id)
            if not text_file:
                continue
            
            # Load content
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine category from path
                category = text_file.parent.name.replace('_', ' ').title()
                
                # Create document
                doc = CurriculumDocument(
                    text_id=text_id,
                    category=category,
                    title=self._format_title(text_id),
                    content=content[:50000],  # Limit content size
                    metadata={'file_path': str(text_file)}
                )
                
                doc_idx = len(self.documents)
                self.documents.append(doc)
                
                # Index by category
                self.category_index[category].append(doc_idx)
                
                # Create chunks for better retrieval
                self._create_chunks(doc, doc_idx)
                
                # Index keywords
                self._index_keywords(doc, doc_idx)
                
                documents_loaded += 1
                
            except Exception as e:
                logger.warning(f"[CURRICULUM-RAG] Failed to load {text_id}: {e}")
        
        logger.info(f"[CURRICULUM-RAG] ✓ Indexed {documents_loaded} documents, {len(self.document_chunks)} chunks")
        logger.info(f"[CURRICULUM-RAG] Categories: {list(self.category_index.keys())}")
    
    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 3,
        categories: Optional[List[str]] = None,
        min_relevance: float = 0.3
    ) -> List[KnowledgeRetrieval]:
        """
        Retrieve relevant knowledge from curriculum.
        
        Args:
            query: Query text
            top_k: Number of results to return
            categories: Filter by categories (None = all)
            min_relevance: Minimum relevance threshold
            
        Returns:
            List of knowledge retrievals
        """
        self.retrieval_count += 1
        
        # Extract query keywords
        query_keywords = self._extract_keywords(query.lower())
        
        # Score all chunks
        chunk_scores = []
        for i, (doc, chunk, chunk_idx) in enumerate(self.document_chunks):
            # Filter by category if specified
            if categories and doc.category not in categories:
                continue
            
            # Calculate relevance score
            score = self._calculate_relevance(query_keywords, chunk.lower(), doc)
            
            if score >= min_relevance:
                chunk_scores.append((i, score))
        
        # Sort by score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        results = []
        seen_docs = set()
        
        for chunk_idx, score in chunk_scores[:top_k * 2]:  # Get more, filter duplicates
            doc, chunk, _ = self.document_chunks[chunk_idx]
            
            # Avoid duplicate documents (take best chunk per doc)
            if doc.text_id in seen_docs:
                continue
            seen_docs.add(doc.text_id)
            
            # Create excerpt
            excerpt = self._create_excerpt(chunk, query_keywords)
            
            results.append(KnowledgeRetrieval(
                document=doc,
                relevance_score=score,
                excerpt=excerpt
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def augment_prompt_with_knowledge(
        self,
        base_prompt: str,
        knowledge_query: Optional[str] = None,
        top_k: int = 2,
        categories: Optional[List[str]] = None
    ) -> str:
        """
        Augment a prompt with relevant curriculum knowledge.
        
        Args:
            base_prompt: Original prompt
            knowledge_query: Query for knowledge (uses base_prompt if None)
            top_k: Number of knowledge pieces to include
            categories: Filter categories
            
        Returns:
            Augmented prompt with curriculum knowledge
        """
        query = knowledge_query or base_prompt
        
        # Retrieve knowledge
        knowledge = self.retrieve_knowledge(
            query=query,
            top_k=top_k,
            categories=categories
        )
        
        if not knowledge:
            return base_prompt
        
        # Build augmented prompt
        augmentation = "\n\n[RELEVANT ACADEMIC KNOWLEDGE]:\n"
        for i, k in enumerate(knowledge, 1):
            augmentation += f"\n{i}. {k.document.title} ({k.document.category}):\n"
            augmentation += f"   {k.excerpt}\n"
        
        return f"{base_prompt}{augmentation}"
    
    def get_category_knowledge(
        self,
        category: str,
        max_docs: int = 5
    ) -> List[CurriculumDocument]:
        """
        Get documents from a specific category.
        
        Args:
            category: Category name
            max_docs: Maximum documents to return
            
        Returns:
            List of documents
        """
        doc_indices = self.category_index.get(category, [])
        return [self.documents[i] for i in doc_indices[:max_docs]]
    
    def _find_text_file(self, text_id: str) -> Optional[Path]:
        """Find text file by ID."""
        # Search all category directories
        for category_dir in self.curriculum_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            for text_file in category_dir.glob("*.txt"):
                if text_id in text_file.stem:
                    return text_file
        
        return None
    
    def _format_title(self, text_id: str) -> str:
        """Format text ID into readable title."""
        # Convert underscores to spaces and capitalize
        parts = text_id.split('_')
        return ' '.join(word.capitalize() for word in parts)
    
    def _create_chunks(self, doc: CurriculumDocument, doc_idx: int):
        """Create text chunks for better retrieval."""
        content = doc.content
        
        # Split into chunks
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            self.document_chunks.append((doc, chunk, i // self.chunk_size))
    
    def _index_keywords(self, doc: CurriculumDocument, doc_idx: int):
        """Index important keywords from document."""
        # Extract significant words (simple approach)
        words = doc.title.lower().split() + doc.category.lower().split()
        
        for word in words:
            if len(word) > 3:  # Skip short words
                self.keyword_index[word].append(doc_idx)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter stop words and short words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for'}
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        return keywords[:20]  # Limit keywords
    
    def _calculate_relevance(
        self,
        query_keywords: List[str],
        chunk_text: str,
        doc: CurriculumDocument
    ) -> float:
        """Calculate relevance score between query and chunk."""
        score = 0.0
        
        # Keyword matching
        chunk_lower = chunk_text.lower()
        matches = sum(1 for kw in query_keywords if kw in chunk_lower)
        keyword_score = matches / max(1, len(query_keywords))
        
        # Title/category bonus
        title_lower = doc.title.lower()
        category_lower = doc.category.lower()
        title_matches = sum(1 for kw in query_keywords if kw in title_lower)
        category_matches = sum(1 for kw in query_keywords if kw in category_lower)
        
        title_bonus = title_matches * 0.2
        category_bonus = category_matches * 0.1
        
        score = keyword_score + title_bonus + category_bonus
        
        return min(1.0, score)
    
    def _create_excerpt(self, chunk: str, keywords: List[str], context_chars: int = 400) -> str:
        """Create relevant excerpt from chunk."""
        chunk_lower = chunk.lower()
        
        # Find best position (most keyword matches)
        best_pos = 0
        best_score = 0
        
        for i in range(0, len(chunk_lower) - context_chars, context_chars // 2):
            window = chunk_lower[i:i + context_chars]
            score = sum(1 for kw in keywords if kw in window)
            if score > best_score:
                best_score = score
                best_pos = i
        
        # Extract excerpt
        excerpt = chunk[best_pos:best_pos + context_chars]
        
        # Clean up
        if best_pos > 0:
            excerpt = "..." + excerpt
        if best_pos + context_chars < len(chunk):
            excerpt = excerpt + "..."
        
        return excerpt.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        return {
            'documents_indexed': len(self.documents),
            'chunks_created': len(self.document_chunks),
            'categories': list(self.category_index.keys()),
            'category_counts': {k: len(v) for k, v in self.category_index.items()},
            'retrievals_performed': self.retrieval_count,
            'keywords_indexed': len(self.keyword_index)
        }


# Predefined category mappings for common queries
CATEGORY_MAPPINGS = {
    'strategy': ['Philosophy Of Science', 'Logic & Reasoning', 'Political Theory'],
    'ethics': ['Ethics & Moral Philosophy', 'Religion & Theology', 'Philosophy'],
    'science': ['Natural Sciences', 'Advanced Sciences', 'More Science'],
    'psychology': ['Psychology', 'Social Sciences', 'Anthropology & Sociology'],
    'math': ['Mathematics', 'Advanced Mathematics', 'Logic & Reasoning'],
    'literature': ['Literature', 'World Literature', 'Poetry', 'Drama'],
    'history': ['History', 'Biography', 'Ancient Classics'],
    'creativity': ['Art & Aesthetics', 'Aesthetics & Art Theory', 'Poetry', 'Literature']
}
