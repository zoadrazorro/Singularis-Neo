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
    """Represents a single document from the university curriculum knowledge base.

    Attributes:
        text_id: A unique identifier for the text.
        category: The academic category (e.g., 'Philosophy', 'Natural Sciences').
        title: The title of the document.
        content: The full text content of the document.
        embedding: An optional numpy array for the document's semantic embedding.
        metadata: A dictionary for any additional metadata, like file path.
    """
    text_id: str
    category: str
    title: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


@dataclass
class KnowledgeRetrieval:
    """Represents a piece of retrieved knowledge from the curriculum.

    Attributes:
        document: The CurriculumDocument from which the knowledge was retrieved.
        relevance_score: A float score indicating how relevant the document is to the query.
        excerpt: A string containing the most relevant snippet of text from the document.
    """
    document: CurriculumDocument
    relevance_score: float
    excerpt: str


class CurriculumRAG:
    """A Retrieval-Augmented Generation system for university curriculum knowledge.

    This class loads, indexes, and retrieves information from a structured
    collection of academic texts. It uses keyword matching and optional
    semantic embeddings to find relevant knowledge to augment AI decision-making.
    """
    
    def __init__(
        self,
        curriculum_path: str = "university_curriculum",
        max_documents: int = 200,
        chunk_size: int = 2000,
        use_embeddings: bool = False
    ):
        """Initializes the CurriculumRAG system.

        Args:
            curriculum_path: The file path to the root of the curriculum directory.
            max_documents: The maximum number of documents to load and index.
            chunk_size: The size of text chunks to split documents into for retrieval.
            use_embeddings: Whether to use sentence-transformer embeddings for similarity.
                            (Note: Requires the 'sentence-transformers' library).
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
        """Loads and indexes the curriculum documents from the specified path.

        Reads a manifest file, then loads, chunks, and indexes the text documents
        by category and keywords.
        """
        logger.info("[CURRICULUM-RAG] Loading university curriculum...")
        
        # Load manifest
        manifest_path = self.curriculum_path / "curriculum_manifest.json"
        if not manifest_path.exists():
            logger.warning(f"[CURRICULUM-RAG] Manifest not found: {manifest_path}")
            return
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        completed = manifest.get('completed', [])
        
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
        """Retrieves the most relevant knowledge from the curriculum based on a query.

        Args:
            query: The text query to search for.
            top_k: The maximum number of knowledge retrievals to return.
            categories: An optional list of categories to restrict the search to.
            min_relevance: The minimum relevance score for a result to be included.

        Returns:
            A list of KnowledgeRetrieval objects, sorted by relevance.
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
        """Augments a given prompt with knowledge retrieved from the curriculum.

        Args:
            base_prompt: The original prompt to be augmented.
            knowledge_query: The query to use for retrieval. If None, the
                             base_prompt is used as the query.
            top_k: The number of knowledge snippets to include.
            categories: An optional list of categories to restrict the search to.

        Returns:
            The augmented prompt string with a "[RELEVANT ACADEMIC KNOWLEDGE]" section.
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
        """Retrieves all documents belonging to a specific category.

        Args:
            category: The name of the category to retrieve.
            max_docs: The maximum number of documents to return from that category.

        Returns:
            A list of CurriculumDocument objects from the specified category.
        """
        doc_indices = self.category_index.get(category, [])
        return [self.documents[i] for i in doc_indices[:max_docs]]
    
    def _find_text_file(self, text_id: str) -> Optional[Path]:
        """Finds the text file corresponding to a given text ID.

        Args:
            text_id: The ID of the text to find.

        Returns:
            A Path object to the found file, or None if not found.
        """
        # Search all category directories
        for category_dir in self.curriculum_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            for text_file in category_dir.glob("*.txt"):
                if text_id in text_file.stem:
                    return text_file
        
        return None
    
    def _format_title(self, text_id: str) -> str:
        """Formats a text_id string into a human-readable title.

        Args:
            text_id: The text ID string (e.g., 'philosophy_of_science').

        Returns:
            A formatted title string (e.g., 'Philosophy Of Science').
        """
        # Convert underscores to spaces and capitalize
        parts = text_id.split('_')
        return ' '.join(word.capitalize() for word in parts)
    
    def _create_chunks(self, doc: CurriculumDocument, doc_idx: int):
        """Splits a document's content into smaller chunks and stores them.

        Args:
            doc: The CurriculumDocument to process.
            doc_idx: The index of the document in the main documents list.
        """
        content = doc.content
        
        # Split into chunks
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            self.document_chunks.append((doc, chunk, i // self.chunk_size))
    
    def _index_keywords(self, doc: CurriculumDocument, doc_idx: int):
        """Extracts and indexes keywords from a document's title and category.

        Args:
            doc: The CurriculumDocument to process.
            doc_idx: The index of the document in the main documents list.
        """
        # Extract significant words (simple approach)
        words = doc.title.lower().split() + doc.category.lower().split()
        
        for word in words:
            if len(word) > 3:  # Skip short words
                self.keyword_index[word].append(doc_idx)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts a list of meaningful keywords from a text string.

        Args:
            text: The text to extract keywords from.

        Returns:
            A list of keyword strings.
        """
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
        """Calculates a relevance score between a query and a text chunk.

        The score is based on keyword matching, with bonuses for matches in the
        document's title and category.

        Args:
            query_keywords: A list of keywords from the user query.
            chunk_text: The text of the document chunk to score.
            doc: The document the chunk belongs to.

        Returns:
            A relevance score between 0.0 and 1.0.
        """
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
        """Creates a relevant excerpt from a chunk centered around query keywords.

        Args:
            chunk: The text chunk to extract from.
            keywords: The list of query keywords to focus on.
            context_chars: The desired number of characters for the excerpt.

        Returns:
            A formatted excerpt string.
        """
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
    
    def get_random_academic_thought(self) -> Optional[KnowledgeRetrieval]:
        """Retrieves a random academic thought using a simulated Brownian motion.

        This method simulates a spontaneous memory recall by performing a random
        walk through the indexed document chunks.

        Returns:
            A KnowledgeRetrieval object with a random excerpt, or None if no
            documents are loaded.
        """
        if not self.document_chunks:
            return None
        
        import random
        import numpy as np
        
        # Brownian motion parameters
        # Start from a random position in knowledge space
        current_pos = random.randint(0, len(self.document_chunks) - 1)
        
        # Random walk for 3-7 steps (Brownian motion)
        walk_steps = random.randint(3, 7)
        
        for _ in range(walk_steps):
            # Brownian step: normally distributed random walk
            # σ controls diffusion rate (how far we can jump)
            step_size = int(np.random.normal(0, len(self.document_chunks) * 0.1))
            current_pos = (current_pos + step_size) % len(self.document_chunks)
        
        # Retrieve the document we landed on
        doc, chunk, chunk_idx = self.document_chunks[current_pos]
        
        # Create a natural-sounding excerpt (first ~200 chars of chunk)
        excerpt = chunk[:200].strip()
        if len(chunk) > 200:
            # Try to end at sentence boundary
            last_period = excerpt.rfind('.')
            if last_period > 100:
                excerpt = excerpt[:last_period + 1]
            else:
                excerpt = excerpt + "..."
        
        # Assign a random "relevance" score (represents how vivid the memory is)
        # Some random thoughts are clearer than others
        vividness = random.uniform(0.3, 0.8)
        
        logger.debug(f"[BROWNIAN-THOUGHT] Random academic thought from {doc.category}: {doc.title}")
        
        return KnowledgeRetrieval(
            document=doc,
            relevance_score=vividness,
            excerpt=excerpt
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the indexed curriculum.

        Returns:
            A dictionary containing statistics like the number of indexed
            documents, chunks, categories, and retrievals performed.
        """
        return {
            'documents_indexed': len(self.documents),
            'chunks_created': len(self.document_chunks),
            'categories': list(self.category_index.keys()),
            'category_counts': {k: len(v) for k, v in self.category_index.items()},
            'retrievals_performed': self.retrieval_count,
            'keywords_indexed': len(self.keyword_index)
        }


# Predefined category mappings for common queries to simplify targeted retrieval.
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
