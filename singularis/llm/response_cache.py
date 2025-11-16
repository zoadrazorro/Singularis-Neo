"""
LLM Response Caching System for Singularis.

Caches common scene→action patterns to reduce redundant LLM queries.
Uses TTL-based expiration, similarity matching, and optional FAISS for semantic search.
"""

import hashlib
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict
from loguru import logger
import numpy as np

# Try to import FAISS for semantic similarity
FAISS_AVAILABLE = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
    logger.info("FAISS available for LLM response cache semantic search")
except ImportError:
    logger.debug("FAISS not available for cache, using basic similarity matching")


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    response: Any
    timestamp: float
    hits: int = 0
    scene_type: str = ""
    health_bucket: str = ""  # low/medium/high
    combat_state: bool = False
    embedding: Optional[np.ndarray] = None  # Semantic embedding for FAISS search
    context_text: str = ""  # Original text for embedding


class LLMResponseCache:
    """
    An intelligent cache for LLM responses, designed to reduce redundant
    queries by storing and retrieving common scene-action patterns.

    This cache supports TTL-based expiration, LRU eviction, and optional
    similarity matching using FAISS for semantic search.
    """
    
    def __init__(
        self,
        max_size: int = 200,
        ttl_seconds: float = 120.0,
        enable_similarity: bool = True
    ):
        """
        Initializes the LLMResponseCache.

        Args:
            max_size (int, optional): The maximum number of entries in the cache.
                                    Defaults to 200.
            ttl_seconds (float, optional): The time-to-live for cache entries in
                                           seconds. Defaults to 120.0.
            enable_similarity (bool, optional): If True, enables similarity matching
                                                for cache lookups. Defaults to True.
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.enable_similarity = enable_similarity
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        
        # FAISS semantic search
        self.use_faiss = FAISS_AVAILABLE and enable_similarity
        if self.use_faiss:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.faiss_index = None
            self.faiss_keys: List[str] = []
            logger.info("LLM Cache: FAISS semantic search enabled")
        
        logger.info(
            f"LLM Response Cache initialized: max_size={max_size}, "
            f"ttl={ttl_seconds}s, similarity={enable_similarity}, faiss={self.use_faiss}"
        )
    
    def _make_key(
        self,
        scene_type: str,
        health: float,
        in_combat: bool,
        available_actions: Tuple[str, ...],
        context_hash: Optional[str] = None
    ) -> str:
        """
        Generate cache key from game state.
        
        Args:
            scene_type: Scene classification
            health: Health percentage (0-100)
            in_combat: Combat state
            available_actions: Tuple of available actions
            context_hash: Optional hash of additional context
            
        Returns:
            Cache key string
        """
        # Bucket health into categories
        if health < 30:
            health_bucket = "low"
        elif health < 70:
            health_bucket = "medium"
        else:
            health_bucket = "high"
        
        # Create key from state components
        actions_str = "|".join(sorted(available_actions))
        
        if context_hash:
            key_parts = [scene_type, health_bucket, str(in_combat), actions_str, context_hash]
        else:
            key_parts = [scene_type, health_bucket, str(in_combat), actions_str]
        
        key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
        return key
    
    def _health_bucket(self, health: float) -> str:
        """Categorize health level."""
        if health < 30:
            return "low"
        elif health < 70:
            return "medium"
        else:
            return "high"
    
    def get(
        self,
        scene_type: str,
        health: float,
        in_combat: bool,
        available_actions: Tuple[str, ...],
        context_hash: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieves a response from the cache.

        This method first attempts an exact match and then falls back to
        similarity matching if enabled.

        Args:
            scene_type (str): The classification of the scene.
            health (float): The player's health.
            in_combat (bool): The combat state.
            available_actions (Tuple[str, ...]): The available actions.
            context_hash (Optional[str], optional): An optional hash of
                                                  additional context. Defaults to None.

        Returns:
            Optional[Any]: The cached response, or None if no match is found.
        """
        # Try exact match first
        key = self._make_key(scene_type, health, in_combat, available_actions, context_hash)
        
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            age = time.time() - entry.timestamp
            if age < self.ttl:
                # Move to end (LRU)
                self._cache.move_to_end(key)
                entry.hits += 1
                self._hits += 1
                
                logger.debug(
                    f"Cache HIT: {scene_type}/{self._health_bucket(health)}/combat={in_combat} "
                    f"(age={age:.1f}s, hits={entry.hits})"
                )
                return entry.response
            else:
                # Expired, remove
                del self._cache[key]
                logger.debug(f"Cache entry expired: age={age:.1f}s > ttl={self.ttl}s")
        
        # Try similarity matching if enabled
        if self.enable_similarity:
            similar = self._find_similar(scene_type, health, in_combat)
            if similar:
                self._hits += 1
                logger.debug(
                    f"Cache SIMILAR HIT: {scene_type}/{self._health_bucket(health)} "
                    f"matched {similar.scene_type}/{similar.health_bucket}"
                )
                return similar.response
        
        self._misses += 1
        logger.debug(f"Cache MISS: {scene_type}/{self._health_bucket(health)}/combat={in_combat}")
        return None
    
    def _find_similar(
        self,
        scene_type: str,
        health: float,
        in_combat: bool
    ) -> Optional[CacheEntry]:
        """
        Find similar cache entry using FAISS or fuzzy matching.
        
        Args:
            scene_type: Target scene type
            health: Target health
            in_combat: Target combat state
            
        Returns:
            Similar entry or None
        """
        # Try FAISS semantic search first
        if self.use_faiss and self.faiss_index is not None and len(self.faiss_keys) > 0:
            try:
                # Create query embedding
                query_text = f"scene:{scene_type} health:{health:.0f} combat:{in_combat}"
                query_embedding = self.embedder.encode([query_text])[0].astype('float32')
                query_embedding = query_embedding.reshape(1, -1)
                
                # Search top 3 similar entries
                k = min(3, len(self.faiss_keys))
                distances, indices = self.faiss_index.search(query_embedding, k)
                
                # Return first valid non-expired entry
                for idx, distance in zip(indices[0], distances[0]):
                    if idx >= 0 and idx < len(self.faiss_keys):
                        key = self.faiss_keys[idx]
                        if key in self._cache:
                            entry = self._cache[key]
                            age = time.time() - entry.timestamp
                            if age < self.ttl:
                                logger.debug(f"FAISS match: distance={distance:.3f}")
                                return entry
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")
        
        # Fallback to basic similarity matching
        health_bucket = self._health_bucket(health)
        
        # Look for entries with same scene + health bucket + combat state
        for entry in reversed(self._cache.values()):
            age = time.time() - entry.timestamp
            if age >= self.ttl:
                continue
                
            if (entry.scene_type == scene_type and
                entry.health_bucket == health_bucket and
                entry.combat_state == in_combat):
                return entry
        
        return None
    
    def put(
        self,
        scene_type: str,
        health: float,
        in_combat: bool,
        available_actions: Tuple[str, ...],
        response: Any,
        context_hash: Optional[str] = None
    ):
        """
        Stores a response in the cache.

        This method handles LRU eviction and, if FAISS is enabled, creates and
        stores a semantic embedding for the entry.

        Args:
            scene_type (str): The classification of the scene.
            health (float): The player's health.
            in_combat (bool): The combat state.
            available_actions (Tuple[str, ...]): The available actions.
            response (Any): The response to cache.
            context_hash (Optional[str], optional): An optional hash of
                                                  additional context. Defaults to None.
        """
        key = self._make_key(scene_type, health, in_combat, available_actions, context_hash)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            # Rebuild FAISS index after eviction
            if self.use_faiss:
                self._rebuild_faiss_index()
            logger.debug(f"Cache evicted oldest entry (LRU)")
        
        # Create context text and embedding for FAISS
        context_text = f"scene:{scene_type} health:{health:.0f} combat:{in_combat}"
        embedding = None
        
        if self.use_faiss:
            try:
                embedding = self.embedder.encode([context_text])[0].astype('float32')
            except Exception as e:
                logger.warning(f"Failed to create embedding: {e}")
        
        # Store entry
        entry = CacheEntry(
            key=key,
            response=response,
            timestamp=time.time(),
            scene_type=scene_type,
            health_bucket=self._health_bucket(health),
            combat_state=in_combat,
            embedding=embedding,
            context_text=context_text
        )
        
        self._cache[key] = entry
        
        # Update FAISS index
        if self.use_faiss and embedding is not None:
            self._add_to_faiss_index(key, embedding)
        
        logger.debug(
            f"Cache PUT: {scene_type}/{entry.health_bucket}/combat={in_combat} "
            f"(size={len(self._cache)}/{self.max_size})"
        )
    
    def _add_to_faiss_index(self, key: str, embedding: np.ndarray):
        """Add embedding to FAISS index."""
        try:
            if self.faiss_index is None:
                # Initialize FAISS index
                dimension = len(embedding)
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_keys = []
            
            # Add to index
            self.faiss_index.add(embedding.reshape(1, -1))
            self.faiss_keys.append(key)
            
        except Exception as e:
            logger.warning(f"Failed to add to FAISS index: {e}")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from current cache entries."""
        if not self.use_faiss:
            return
        
        try:
            embeddings = []
            keys = []
            
            for key, entry in self._cache.items():
                if entry.embedding is not None:
                    embeddings.append(entry.embedding)
                    keys.append(key)
            
            if embeddings:
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatL2(dimension)
                embeddings_array = np.array(embeddings).astype('float32')
                self.faiss_index.add(embeddings_array)
                self.faiss_keys = keys
                logger.debug(f"FAISS index rebuilt with {len(keys)} entries")
            else:
                self.faiss_index = None
                self.faiss_keys = []
                
        except Exception as e:
            logger.warning(f"Failed to rebuild FAISS index: {e}")
    
    def clear(self):
        """Clears all entries from the cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the cache.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl
        }
    
    def prune_expired(self):
        """Removes all expired entries from the cache."""
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if (now - entry.timestamp) >= self.ttl
        ]
        
        for key in expired:
            del self._cache[key]
        
        if expired:
            logger.debug(f"Pruned {len(expired)} expired cache entries")
