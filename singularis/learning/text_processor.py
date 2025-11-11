"""
Text Processing for Consciousness Learning

Processes philosophy texts and curriculum through the Singularis consciousness
engine, enabling Hebbian learning and knowledge integration.

From ETHICA UNIVERSALIS Part II:
"The more the mind perceives, the greater its power of understanding."
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass
from loguru import logger
import re


@dataclass
class TextChunk:
    """A chunk of text for processing."""
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: Dict


class TextProcessor:
    """
    Process large texts into manageable chunks for LLM processing.
    
    Philosophy:
    Each text is a MODE of Being expressing knowledge.
    Chunking preserves semantic coherence while fitting context windows.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 3000,  # ~750 tokens at 4 chars/token
        overlap: int = 200,  # Overlap for context continuity
    ):
        """
        Initialize text processor.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        logger.info(
            "TextProcessor initialized",
            extra={
                "max_chunk_size": max_chunk_size,
                "overlap": overlap,
            }
        )
    
    def chunk_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict] = None,
    ) -> List[TextChunk]:
        """
        Chunk text into processable segments.
        
        Strategy:
        1. Split on paragraph boundaries when possible
        2. Maintain overlap for context
        3. Preserve semantic units
        
        Args:
            text: Full text to chunk
            source: Source identifier (filename)
            metadata: Optional metadata
            
        Returns:
            List of TextChunk objects
        """
        # Clean text
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        
        # Build chunks
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds limit, save current chunk
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Create TextChunk objects
        total_chunks = len(chunks)
        text_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            text_chunks.append(TextChunk(
                text=chunk_text,
                source=source,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=metadata or {},
            ))
        
        logger.info(
            f"Chunked text from {source}",
            extra={
                "total_chunks": total_chunks,
                "avg_chunk_size": sum(len(c.text) for c in text_chunks) / total_chunks,
            }
        )
        
        return text_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers and headers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\[Page \d+\]', '', text)
        
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = text.split('\n\n')
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def load_text_file(self, filepath: Path) -> str:
        """Load text file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {filepath}")
    
    def process_directory(
        self,
        directory: Path,
        file_pattern: str = "*.txt",
    ) -> Iterator[Tuple[str, List[TextChunk]]]:
        """
        Process all text files in a directory.
        
        Args:
            directory: Directory to process
            file_pattern: Glob pattern for files
            
        Yields:
            (filename, chunks) tuples
        """
        files = sorted(directory.glob(file_pattern))
        
        logger.info(
            f"Processing directory: {directory}",
            extra={"file_count": len(files)}
        )
        
        for filepath in files:
            try:
                text = self.load_text_file(filepath)
                chunks = self.chunk_text(
                    text=text,
                    source=filepath.name,
                    metadata={
                        "filepath": str(filepath),
                        "size_bytes": len(text),
                    }
                )
                yield filepath.name, chunks
                
            except Exception as e:
                logger.error(
                    f"Failed to process {filepath.name}",
                    extra={"error": str(e)}
                )
                continue


class CurriculumLoader:
    """
    Load and organize university curriculum texts.
    
    Philosophy:
    Knowledge is organized hierarchically.
    Each domain contributes to the whole.
    """
    
    def __init__(self, curriculum_root: Path):
        """
        Initialize curriculum loader.
        
        Args:
            curriculum_root: Root directory of curriculum
        """
        self.curriculum_root = Path(curriculum_root)
        self.manifest_path = self.curriculum_root / "curriculum_manifest.json"
        self.manifest = self._load_manifest()
        
        logger.info(
            "CurriculumLoader initialized",
            extra={"root": str(curriculum_root)}
        )
    
    def _load_manifest(self) -> Optional[Dict]:
        """Load curriculum manifest if it exists."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_domains(self) -> List[str]:
        """Get list of curriculum domains."""
        domains = []
        for item in self.curriculum_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                domains.append(item.name)
        return sorted(domains)
    
    def get_domain_files(self, domain: str) -> List[Path]:
        """Get all text files in a domain."""
        domain_path = self.curriculum_root / domain
        if not domain_path.exists():
            return []
        
        files = []
        for ext in ['*.txt', '*.md']:
            files.extend(domain_path.glob(ext))
        
        return sorted(files)
    
    def iterate_curriculum(
        self,
        processor: TextProcessor,
        domains: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, str, List[TextChunk]]]:
        """
        Iterate through curriculum, yielding chunks.
        
        Args:
            processor: TextProcessor instance
            domains: Optional list of domains to process (None = all)
            
        Yields:
            (domain, filename, chunks) tuples
        """
        if domains is None:
            domains = self.get_domains()
        
        for domain in domains:
            logger.info(f"Processing domain: {domain}")
            
            files = self.get_domain_files(domain)
            
            for filepath in files:
                try:
                    text = processor.load_text_file(filepath)
                    chunks = processor.chunk_text(
                        text=text,
                        source=filepath.name,
                        metadata={
                            "domain": domain,
                            "filepath": str(filepath),
                            "size_bytes": len(text),
                        }
                    )
                    yield domain, filepath.name, chunks
                    
                except Exception as e:
                    logger.error(
                        f"Failed to process {filepath.name} in {domain}",
                        extra={"error": str(e)}
                    )
                    continue


class LearningProgress:
    """
    Track learning progress across texts.
    
    Philosophy:
    Learning is accumulation of adequate ideas.
    Progress is measured by coherence increase.
    """
    
    def __init__(self, progress_file: Path):
        """
        Initialize learning progress tracker.
        
        Args:
            progress_file: Path to progress JSON file
        """
        self.progress_file = Path(progress_file)
        self.progress = self._load_progress()
        
        logger.info(
            "LearningProgress initialized",
            extra={"progress_file": str(progress_file)}
        )
    
    def _load_progress(self) -> Dict:
        """Load existing progress or create new."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "texts_processed": [],
            "chunks_processed": 0,
            "total_coherentia": 0.0,
            "avg_coherentia": 0.0,
            "ethical_actions": 0,
            "total_processing_time_ms": 0.0,
        }
    
    def mark_processed(
        self,
        source: str,
        chunk_index: int,
        coherentia: float,
        processing_time_ms: float,
        ethical: bool,
    ):
        """Mark a chunk as processed."""
        self.progress["chunks_processed"] += 1
        self.progress["total_coherentia"] += coherentia
        self.progress["avg_coherentia"] = (
            self.progress["total_coherentia"] / self.progress["chunks_processed"]
        )
        self.progress["total_processing_time_ms"] += processing_time_ms
        
        if ethical:
            self.progress["ethical_actions"] += 1
        
        # Track unique texts
        if source not in self.progress["texts_processed"]:
            self.progress["texts_processed"].append(source)
    
    def save(self):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        
        logger.info(
            "Progress saved",
            extra={
                "chunks_processed": self.progress["chunks_processed"],
                "avg_coherentia": self.progress["avg_coherentia"],
            }
        )
    
    def get_stats(self) -> Dict:
        """Get progress statistics."""
        return {
            "texts_processed": len(self.progress["texts_processed"]),
            "chunks_processed": self.progress["chunks_processed"],
            "avg_coherentia": self.progress["avg_coherentia"],
            "ethical_rate": (
                self.progress["ethical_actions"] / max(1, self.progress["chunks_processed"])
            ),
            "total_time_hours": self.progress["total_processing_time_ms"] / (1000 * 60 * 60),
        }
