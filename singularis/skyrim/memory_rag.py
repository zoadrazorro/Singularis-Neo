"""
RAG (Retrieval-Augmented Generation) for Perceptual and Cognitive Memories

Enhances planning and decision-making by retrieving relevant past experiences:
1. Stores perceptual memories (what was seen/experienced)
2. Stores cognitive memories (decisions made, outcomes)
3. Retrieves similar experiences using embedding similarity
4. Augments LLM context with relevant memories

Philosophical grounding:
- Memory enables learning from experience
- Similar contexts suggest similar actions
- Past successes guide future decisions
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import time


@dataclass
class PerceptualMemory:
    """Represents a memory of a perceived scene or event.

    Attributes:
        visual_embedding: A CLIP embedding vector of the visual input.
        scene_type: The classified type of the scene (e.g., 'dungeon', 'city').
        location: The in-game location where the perception occurred.
        timestamp: The time the memory was recorded.
        context: A dictionary of additional context about the perception.
    """
    visual_embedding: np.ndarray
    scene_type: str
    location: str
    timestamp: float
    context: Dict[str, Any]


@dataclass
class CognitiveMemory:
    """Represents a memory of a decision-making process and its outcome.

    Attributes:
        situation: A dictionary describing the situation that prompted the decision.
        action_taken: The action that the agent chose to take.
        outcome: A dictionary describing the result of the action.
        success: A boolean indicating whether the outcome was considered a success.
        reasoning: The reasoning or justification for the action taken.
        timestamp: The time the memory was recorded.
    """
    situation: Dict[str, Any]
    action_taken: str
    outcome: Dict[str, Any]
    success: bool
    reasoning: str
    timestamp: float


class MemoryRAG:
    """A Retrieval-Augmented Generation (RAG) system for managing game memories.

    This system stores and retrieves two types of memories:
    1.  **Perceptual Memories:** What the agent has seen (using visual embeddings).
    2.  **Cognitive Memories:** What the agent has done and what the outcomes were.

    It retrieves relevant past experiences to augment the context provided to the
    LLM, enabling more informed, experience-based decision-making.
    """
    
    def __init__(
        self,
        perceptual_capacity: int = 1000,
        cognitive_capacity: int = 500
    ):
        """Initializes the MemoryRAG system.

        Args:
            perceptual_capacity: The maximum number of perceptual memories to store.
            cognitive_capacity: The maximum number of cognitive memories to store.
        """
        self.perceptual_capacity = perceptual_capacity
        self.cognitive_capacity = cognitive_capacity
        
        # Memory stores
        self.perceptual_memories: deque = deque(maxlen=perceptual_capacity)
        self.cognitive_memories: deque = deque(maxlen=cognitive_capacity)
        
        # Indexing for fast retrieval
        self.perceptual_embeddings: List[np.ndarray] = []
        self.cognitive_embeddings: List[np.ndarray] = []
        
        print("[RAG] Memory RAG system initialized")
    
    def store_perceptual_memory(
        self,
        visual_embedding: np.ndarray,
        scene_type: str,
        location: str,
        context: Dict[str, Any]
    ):
        """Stores a new perceptual memory.

        Args:
            visual_embedding: The CLIP embedding of the perceived visual scene.
            scene_type: The classified type of the scene.
            location: The in-game location.
            context: A dictionary of additional contextual information.
        """
        memory = PerceptualMemory(
            visual_embedding=visual_embedding,
            scene_type=scene_type,
            location=location,
            timestamp=time.time(),
            context=context
        )
        
        self.perceptual_memories.append(memory)
        self.perceptual_embeddings.append(visual_embedding)
        
        # Trim embeddings if needed
        if len(self.perceptual_embeddings) > self.perceptual_capacity:
            self.perceptual_embeddings.pop(0)
    
    def store_cognitive_memory(
        self,
        situation: Dict[str, Any],
        action_taken: str,
        outcome: Dict[str, Any],
        success: bool,
        reasoning: str = ""
    ):
        """Stores a new cognitive memory of a decision and its outcome.

        Args:
            situation: The situation that prompted the decision.
            action_taken: The action that was performed.
            outcome: The outcome resulting from the action.
            success: Whether the outcome was successful.
            reasoning: The reasoning behind the decision.
        """
        memory = CognitiveMemory(
            situation=situation,
            action_taken=action_taken,
            outcome=outcome,
            success=success,
            reasoning=reasoning,
            timestamp=time.time()
        )
        
        self.cognitive_memories.append(memory)
        
        # Create embedding from situation
        situation_embedding = self._create_situation_embedding(situation)
        self.cognitive_embeddings.append(situation_embedding)
        
        # Trim embeddings if needed
        if len(self.cognitive_embeddings) > self.cognitive_capacity:
            self.cognitive_embeddings.pop(0)
    
    def retrieve_similar_perceptions(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[PerceptualMemory, float]]:
        """Retrieves perceptual memories that are visually similar to a query embedding.

        Args:
            query_embedding: The visual embedding of the current scene to match against.
            top_k: The maximum number of memories to return.
            similarity_threshold: The minimum cosine similarity for a memory to be considered.

        Returns:
            A list of tuples, each containing a PerceptualMemory and its
            similarity score, sorted by similarity.
        """
        if not self.perceptual_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.perceptual_embeddings):
            similarity = self._cosine_similarity(query_embedding, emb)
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        results = []
        for idx, sim in similarities[:top_k]:
            if idx < len(self.perceptual_memories):
                results.append((self.perceptual_memories[idx], sim))
        
        return results
    
    def retrieve_similar_decisions(
        self,
        current_situation: Dict[str, Any],
        top_k: int = 3,
        only_successful: bool = False
    ) -> List[Tuple[CognitiveMemory, float]]:
        """Retrieves cognitive memories from situations similar to the current one.

        Args:
            current_situation: A dictionary describing the current situation.
            top_k: The maximum number of memories to return.
            only_successful: If True, only retrieve memories of successful decisions.

        Returns:
            A list of tuples, each containing a CognitiveMemory and its
            similarity score, sorted by similarity.
        """
        if not self.cognitive_embeddings:
            return []
        
        # Create embedding for current situation
        query_embedding = self._create_situation_embedding(current_situation)
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.cognitive_embeddings):
            similarity = self._cosine_similarity(query_embedding, emb)
            
            # Filter by success if requested
            if only_successful:
                if i < len(self.cognitive_memories):
                    if not self.cognitive_memories[i].success:
                        continue
            
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        results = []
        for idx, sim in similarities[:top_k]:
            if idx < len(self.cognitive_memories):
                results.append((self.cognitive_memories[idx], sim))
        
        return results
    
    def augment_context_with_memories(
        self,
        current_visual: np.ndarray,
        current_situation: Dict[str, Any],
        max_memories: int = 3
    ) -> str:
        """Constructs a formatted string of relevant memories to augment an LLM prompt.

        Args:
            current_visual: The visual embedding of the current scene.
            current_situation: A dictionary describing the current situation.
            max_memories: The maximum number of memories (of each type) to include.

        Returns:
            A formatted string containing relevant past perceptions and decisions,
            or an empty string if no relevant memories are found.
        """
        context_parts = []
        
        # Retrieve similar perceptions
        similar_perceptions = self.retrieve_similar_perceptions(
            current_visual,
            top_k=max_memories
        )
        
        if similar_perceptions:
            context_parts.append("\nRELEVANT PAST PERCEPTIONS:")
            for i, (memory, similarity) in enumerate(similar_perceptions, 1):
                context_parts.append(
                    f"{i}. Similar scene ({similarity:.2f} match): "
                    f"{memory.scene_type} at {memory.location}"
                )
        
        # Retrieve similar decisions
        similar_decisions = self.retrieve_similar_decisions(
            current_situation,
            top_k=max_memories,
            only_successful=True
        )
        
        if similar_decisions:
            context_parts.append("\nRELEVANT PAST DECISIONS (successful):")
            for i, (memory, similarity) in enumerate(similar_decisions, 1):
                context_parts.append(
                    f"{i}. Similar situation ({similarity:.2f} match): "
                    f"Action '{memory.action_taken}' → "
                    f"{self._summarize_outcome(memory.outcome)}"
                )
                if memory.reasoning:
                    context_parts.append(f"   Reasoning: {memory.reasoning}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _create_situation_embedding(self, situation: Dict[str, Any]) -> np.ndarray:
        """Creates a simple vector embedding from a situation dictionary.

        Note: This is a basic implementation. A more advanced system would use
        a learned model to create more meaningful embeddings.

        Args:
            situation: The dictionary describing the situation.

        Returns:
            A numpy array representing the situation embedding.
        """
        # Simple hash-based embedding
        # In production, would use learned embeddings
        features = []
        
        # Extract key features
        features.append(hash(situation.get('scene', '')) % 1000 / 1000.0)
        features.append(situation.get('health', 100) / 100.0)
        features.append(float(situation.get('in_combat', False)))
        features.append(hash(situation.get('location', '')) % 1000 / 1000.0)
        
        # Pad to fixed size
        while len(features) < 16:
            features.append(0.0)
        
        return np.array(features[:16])
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates the cosine similarity between two numpy vectors.

        Args:
            a: The first vector.
            b: The second vector.

        Returns:
            The cosine similarity score, clamped between 0.0 and 1.0.
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return max(0.0, min(1.0, dot_product / (norm_a * norm_b)))
    
    def _summarize_outcome(self, outcome: Dict[str, Any]) -> str:
        """Creates a brief, human-readable summary of an outcome dictionary.

        Args:
            outcome: The outcome dictionary to summarize.

        Returns:
            A summary string.
        """
        parts = []
        
        if 'health' in outcome:
            parts.append(f"health={outcome['health']:.0f}")
        
        if 'scene' in outcome:
            parts.append(f"scene={outcome['scene']}")
        
        if 'progress' in outcome:
            parts.append("progress made")
        
        return ", ".join(parts) if parts else "outcome recorded"
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the current state of the memory system.

        Returns:
            A dictionary containing the number and capacity of stored memories.
        """
        return {
            'perceptual_memories': len(self.perceptual_memories),
            'cognitive_memories': len(self.cognitive_memories),
            'total_memories': len(self.perceptual_memories) + len(self.cognitive_memories),
            'perceptual_capacity': self.perceptual_capacity,
            'cognitive_capacity': self.cognitive_capacity
        }
    
    def get_recent_successes(self, n: int = 5) -> List[CognitiveMemory]:
        """Retrieves the most recent successful cognitive memories.

        Args:
            n: The number of recent successes to retrieve.

        Returns:
            A list of the most recent successful CognitiveMemory objects.
        """
        successes = [
            mem for mem in self.cognitive_memories
            if mem.success
        ]
        return list(successes)[-n:]
    
    def clear_old_memories(self, age_threshold_seconds: float = 3600):
        """Removes memories that are older than a specified threshold.

        Args:
            age_threshold_seconds: The maximum age of memories to keep, in seconds.
        """
        current_time = time.time()
        
        # Clear old perceptual memories
        while (self.perceptual_memories and 
               current_time - self.perceptual_memories[0].timestamp > age_threshold_seconds):
            self.perceptual_memories.popleft()
            if self.perceptual_embeddings:
                self.perceptual_embeddings.pop(0)
        
        # Clear old cognitive memories
        while (self.cognitive_memories and 
               current_time - self.cognitive_memories[0].timestamp > age_threshold_seconds):
            self.cognitive_memories.popleft()
            if self.cognitive_embeddings:
                self.cognitive_embeddings.pop(0)
        
        print(f"[RAG] Cleared memories older than {age_threshold_seconds}s")
