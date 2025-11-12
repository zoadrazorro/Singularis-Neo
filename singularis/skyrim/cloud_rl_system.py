"""
Cloud-Enhanced Reinforcement Learning System for Skyrim AGI

Integrates RL with:
- Cloud LLM APIs (Gemini + Claude) for reward shaping and policy guidance
- MoE expert consensus for action evaluation
- Persistent memory with RAG context fetching
- Experience replay with semantic similarity search
- Consciousness-guided learning (Δ𝒞 as reward signal)

Philosophy:
Learning is the increase of coherence (𝒞) through experience.
Cloud LLMs provide high-level strategic guidance while local RL
learns tactical execution patterns.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np

from loguru import logger

# Try ChromaDB first
CHROMADB_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("ChromaDB available for RAG")
except (ImportError, Exception) as e:
    logger.warning(f"ChromaDB not available: {e}")
    
    # Try FAISS as alternative
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        FAISS_AVAILABLE = True
        logger.info("Using FAISS for RAG (ChromaDB unavailable)")
    except ImportError:
        logger.warning("FAISS not available either. Install with: pip install faiss-cpu sentence-transformers")
        logger.info("RAG features will be disabled.")


@dataclass
class Experience:
    """Single RL experience with rich context."""
    # State information
    state_vector: np.ndarray
    state_description: str  # Natural language description
    scene_type: str
    location: str
    health: float
    stamina: float
    magicka: float
    enemies_nearby: int
    in_combat: bool
    
    # Action information
    action: str
    action_type: str
    
    # Outcome information
    reward: float
    next_state_vector: np.ndarray
    next_state_description: str
    done: bool
    
    # Cloud LLM feedback
    llm_evaluation: Optional[str] = None
    llm_reward_adjustment: float = 0.0
    moe_consensus: Optional[str] = None
    moe_coherence: float = 0.0
    
    # Consciousness metrics
    coherence_before: float = 0.0
    coherence_after: float = 0.0
    coherence_delta: float = 0.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    episode_id: int = 0
    step_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'state_description': self.state_description,
            'scene_type': self.scene_type,
            'location': self.location,
            'health': self.health,
            'stamina': self.stamina,
            'magicka': self.magicka,
            'enemies_nearby': self.enemies_nearby,
            'in_combat': self.in_combat,
            'action': self.action,
            'action_type': self.action_type,
            'reward': self.reward,
            'next_state_description': self.next_state_description,
            'done': self.done,
            'llm_evaluation': self.llm_evaluation,
            'llm_reward_adjustment': self.llm_reward_adjustment,
            'moe_consensus': self.moe_consensus,
            'moe_coherence': self.moe_coherence,
            'coherence_before': self.coherence_before,
            'coherence_after': self.coherence_after,
            'coherence_delta': self.coherence_delta,
            'timestamp': self.timestamp,
            'episode_id': self.episode_id,
            'step_id': self.step_id,
        }


@dataclass
class RLMemoryConfig:
    """Configuration for RL memory system."""
    memory_dir: str = "skyrim_rl_memory"
    max_experiences: int = 100000
    batch_size: int = 32
    
    # RAG settings
    use_rag: bool = True
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    
    # Cloud LLM settings
    use_cloud_reward_shaping: bool = True
    reward_shaping_frequency: int = 10  # Every N experiences
    use_moe_evaluation: bool = True
    
    # Persistence
    save_frequency: int = 100  # Save every N experiences
    auto_save: bool = True


class CloudRLMemory:
    """
    RL Memory system with cloud LLM integration and RAG.
    
    Features:
    - Persistent experience storage
    - Semantic similarity search via ChromaDB
    - Cloud LLM reward shaping
    - MoE consensus integration
    - Consciousness-guided learning
    """
    
    def __init__(
        self,
        config: Optional[RLMemoryConfig] = None,
        hybrid_llm=None,
        moe=None,
    ):
        """Initialize cloud RL memory system."""
        self.config = config or RLMemoryConfig()
        self.hybrid_llm = hybrid_llm
        self.moe = moe
        
        # Create memory directory
        self.memory_path = Path(self.config.memory_dir)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Experience replay buffer
        self.experiences: deque = deque(maxlen=self.config.max_experiences)
        
        # ChromaDB for semantic search
        self.chroma_client = None
        self.collection = None
        if CHROMADB_AVAILABLE and self.config.use_rag:
            self._initialize_chromadb()
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'cloud_evaluations': 0,
            'moe_evaluations': 0,
            'avg_reward': 0.0,
            'avg_coherence_delta': 0.0,
            'successful_actions': 0,
            'failed_actions': 0,
        }
        
        # Load existing experiences
        self._load_experiences()
        
        logger.info(f"Cloud RL Memory initialized: {len(self.experiences)} experiences loaded")
    
    def _initialize_chromadb(self):
        """Initialize RAG backend (ChromaDB or FAISS)."""
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.memory_path / "chromadb"),
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Create or get collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name="skyrim_experiences",
                    metadata={"description": "Skyrim RL experiences with semantic search"}
                )
                
                logger.info("ChromaDB initialized for RAG context fetching")
                return
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.chroma_client = None
                self.collection = None
        
        # Try FAISS as fallback
        if FAISS_AVAILABLE:
            try:
                self._initialize_faiss()
                logger.info("FAISS initialized for RAG context fetching")
                return
            except Exception as e:
                logger.error(f"Failed to initialize FAISS: {e}")
        
        logger.warning("No RAG backend available")
    
    def _initialize_faiss(self):
        """Initialize FAISS for semantic search."""
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_metadata = []  # Store metadata separately
        
        # Try to load existing index
        index_path = self.memory_path / "faiss_index.bin"
        metadata_path = self.memory_path / "faiss_metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    self.faiss_metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
        
        self.collection = "faiss"  # Mark as using FAISS
    
    async def add_experience(
        self,
        experience: Experience,
        request_cloud_evaluation: bool = False
    ):
        """
        Add experience to memory with optional cloud LLM evaluation.
        
        Args:
            experience: Experience to add
            request_cloud_evaluation: Whether to request cloud LLM feedback
        """
        # Cloud LLM reward shaping
        if request_cloud_evaluation and self.config.use_cloud_reward_shaping:
            if self.stats['total_experiences'] % self.config.reward_shaping_frequency == 0:
                await self._cloud_reward_shaping(experience)
        
        # MoE evaluation
        if self.config.use_moe_evaluation and self.moe:
            await self._moe_evaluation(experience)
        
        # Add to replay buffer
        self.experiences.append(experience)
        
        # Add to ChromaDB for semantic search
        if self.collection:
            self._add_to_chromadb(experience)
        
        # Update statistics
        self.stats['total_experiences'] += 1
        self.stats['avg_reward'] = (
            (self.stats['avg_reward'] * (self.stats['total_experiences'] - 1) + experience.reward) /
            self.stats['total_experiences']
        )
        self.stats['avg_coherence_delta'] = (
            (self.stats['avg_coherence_delta'] * (self.stats['total_experiences'] - 1) + experience.coherence_delta) /
            self.stats['total_experiences']
        )
        
        if experience.reward > 0:
            self.stats['successful_actions'] += 1
        else:
            self.stats['failed_actions'] += 1
        
        # Auto-save
        if self.config.auto_save and self.stats['total_experiences'] % self.config.save_frequency == 0:
            self.save()
    
    async def _cloud_reward_shaping(self, experience: Experience):
        """Use cloud LLM to evaluate and shape reward."""
        if not self.hybrid_llm:
            return
        
        try:
            prompt = f"""Evaluate this Skyrim gameplay action:

State: {experience.state_description}
Location: {experience.location}
Health: {experience.health}%, Enemies: {experience.enemies_nearby}
In Combat: {experience.in_combat}

Action Taken: {experience.action}

Outcome: {experience.next_state_description}
Base Reward: {experience.reward}

Evaluate this action on a scale of -1.0 to +1.0:
- Was it strategically sound?
- Did it improve the situation?
- Was it appropriate for the context?

Provide a reward adjustment (-1.0 to +1.0) and brief explanation.
Format: REWARD: <number>
EXPLANATION: <text>"""

            system_prompt = """You are an expert Skyrim strategist evaluating gameplay decisions.
Provide constructive feedback to help the AI learn better strategies."""

            response = await self.hybrid_llm.generate_reasoning(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=256
            )
            
            # Parse response
            reward_adj = 0.0
            explanation = response
            
            if "REWARD:" in response:
                try:
                    reward_line = [l for l in response.split('\n') if 'REWARD:' in l][0]
                    reward_adj = float(reward_line.split('REWARD:')[1].strip().split()[0])
                    reward_adj = max(-1.0, min(1.0, reward_adj))  # Clamp
                except:
                    pass
            
            if "EXPLANATION:" in response:
                try:
                    explanation = response.split('EXPLANATION:')[1].strip()
                except:
                    pass
            
            experience.llm_evaluation = explanation
            experience.llm_reward_adjustment = reward_adj
            
            self.stats['cloud_evaluations'] += 1
            
            logger.debug(f"Cloud reward shaping: {reward_adj:+.2f} - {explanation[:100]}")
            
        except Exception as e:
            logger.error(f"Cloud reward shaping failed: {e}")
    
    async def _moe_evaluation(self, experience: Experience):
        """Use MoE expert consensus to evaluate action."""
        if not self.moe:
            return
        
        try:
            prompt = f"""Evaluate this action in Skyrim:

Situation: {experience.state_description}
Action: {experience.action}
Result: {experience.next_state_description}

Was this a good decision? Rate 0-10 and explain briefly."""

            # Query reasoning experts
            response = await self.moe.query_reasoning_experts(
                prompt=prompt,
                context={
                    'location': experience.location,
                    'health': experience.health,
                    'in_combat': experience.in_combat
                }
            )
            
            experience.moe_consensus = response.consensus
            experience.moe_coherence = response.coherence_score
            
            self.stats['moe_evaluations'] += 1
            
            logger.debug(f"MoE evaluation: coherence={response.coherence_score:.2f}")
            
        except Exception as e:
            logger.error(f"MoE evaluation failed: {e}")
    
    def _add_to_chromadb(self, experience: Experience):
        """Add experience to RAG backend (ChromaDB or FAISS)."""
        if not self.collection:
            return
        
        try:
            # Create document text for embedding
            doc_text = f"""
            State: {experience.state_description}
            Location: {experience.location}
            Action: {experience.action}
            Outcome: {experience.next_state_description}
            Reward: {experience.reward}
            Coherence Delta: {experience.coherence_delta}
            """
            
            if CHROMADB_AVAILABLE and self.collection != "faiss":
                # Add to ChromaDB
                self.collection.add(
                    documents=[doc_text],
                    metadatas=[experience.to_dict()],
                    ids=[f"exp_{experience.episode_id}_{experience.step_id}_{experience.timestamp}"]
                )
            elif FAISS_AVAILABLE and self.collection == "faiss":
                # Add to FAISS
                embedding = self.sentence_model.encode([doc_text])[0]
                self.faiss_index.add(np.array([embedding], dtype=np.float32))
                self.faiss_metadata.append(experience.to_dict())
                
                # Periodically save FAISS index
                if len(self.faiss_metadata) % 100 == 0:
                    self._save_faiss_index()
            
        except Exception as e:
            logger.error(f"Failed to add to RAG backend: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            index_path = self.memory_path / "faiss_index.bin"
            metadata_path = self.memory_path / "faiss_metadata.pkl"
            
            faiss.write_index(self.faiss_index, str(index_path))
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.faiss_metadata, f)
            
            logger.debug(f"Saved FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def get_similar_experiences(
        self,
        query_state: str,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[Experience]:
        """
        Retrieve similar experiences using RAG.
        
        Args:
            query_state: Natural language description of current state
            top_k: Number of similar experiences to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar experiences
        """
        if not self.collection:
            # Fallback to random sampling
            return list(np.random.choice(
                list(self.experiences),
                size=min(top_k or self.config.rag_top_k, len(self.experiences)),
                replace=False
            ))
        
        top_k = top_k or self.config.rag_top_k
        similarity_threshold = similarity_threshold or self.config.rag_similarity_threshold
        
        try:
            if CHROMADB_AVAILABLE and self.collection != "faiss":
                # Query ChromaDB
                results = self.collection.query(
                    query_texts=[query_state],
                    n_results=top_k
                )
                
                # Convert back to Experience objects
                similar_exps = []
                if results['metadatas'] and results['metadatas'][0]:
                    for metadata in results['metadatas'][0]:
                        similar_exps.append(metadata)
                
                logger.debug(f"Retrieved {len(similar_exps)} similar experiences via ChromaDB")
                return similar_exps
            
            elif FAISS_AVAILABLE and self.collection == "faiss":
                # Query FAISS
                if self.faiss_index.ntotal == 0:
                    return []
                
                # Encode query
                query_embedding = self.sentence_model.encode([query_state])[0]
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # Search
                k = min(top_k, self.faiss_index.ntotal)
                distances, indices = self.faiss_index.search(query_vector, k)
                
                # Get metadata for results
                similar_exps = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(self.faiss_metadata):
                        # Convert distance to similarity (lower distance = higher similarity)
                        similarity = 1.0 / (1.0 + dist)
                        if similarity >= similarity_threshold:
                            similar_exps.append(self.faiss_metadata[idx])
                
                logger.debug(f"Retrieved {len(similar_exps)} similar experiences via FAISS")
                return similar_exps
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []
    
    def sample_batch(self, batch_size: int = None) -> List[Experience]:
        """Sample random batch for training."""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.experiences) < batch_size:
            return list(self.experiences)
        
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]
    
    def sample_prioritized_batch(
        self,
        batch_size: int = None,
        prioritize_high_reward: bool = True,
        prioritize_high_coherence: bool = True
    ) -> List[Experience]:
        """Sample batch with prioritization."""
        batch_size = batch_size or self.config.batch_size
        
        if len(self.experiences) < batch_size:
            return list(self.experiences)
        
        # Calculate priorities
        priorities = []
        for exp in self.experiences:
            priority = 1.0
            
            if prioritize_high_reward:
                priority += abs(exp.reward)
            
            if prioritize_high_coherence:
                priority += abs(exp.coherence_delta)
            
            # Boost cloud-evaluated experiences
            if exp.llm_evaluation:
                priority *= 1.5
            
            # Boost MoE-evaluated experiences
            if exp.moe_consensus:
                priority *= 1.3
            
            priorities.append(priority)
        
        # Normalize priorities
        priorities = np.array(priorities)
        priorities = priorities / priorities.sum()
        
        # Sample with priorities
        indices = np.random.choice(
            len(self.experiences),
            batch_size,
            replace=False,
            p=priorities
        )
        
        return [self.experiences[i] for i in indices]
    
    def save(self):
        """Save experiences to disk."""
        try:
            # Save experiences as pickle
            exp_file = self.memory_path / "experiences.pkl"
            with open(exp_file, 'wb') as f:
                pickle.dump(list(self.experiences), f)
            
            # Save statistics as JSON
            stats_file = self.memory_path / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            logger.info(f"Saved {len(self.experiences)} experiences to {self.memory_path}")
            
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
    
    def _load_experiences(self):
        """Load experiences from disk."""
        try:
            exp_file = self.memory_path / "experiences.pkl"
            if exp_file.exists():
                with open(exp_file, 'rb') as f:
                    loaded_exps = pickle.load(f)
                    self.experiences.extend(loaded_exps)
                
                logger.info(f"Loaded {len(loaded_exps)} experiences from disk")
            
            stats_file = self.memory_path / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats.update(json.load(f))
                
                logger.info("Loaded RL statistics from disk")
                
        except Exception as e:
            logger.error(f"Failed to load experiences: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.experiences),
            'buffer_capacity': self.config.max_experiences,
            'buffer_utilization': len(self.experiences) / self.config.max_experiences,
            'rag_enabled': self.collection is not None,
            'cloud_llm_enabled': self.hybrid_llm is not None,
            'moe_enabled': self.moe is not None,
        }
    
    async def close(self):
        """Close and cleanup."""
        self.save()
        logger.info("Cloud RL Memory closed")


class CloudRLAgent:
    """
    RL Agent with cloud LLM guidance and MoE integration.
    
    Combines:
    - Traditional RL (Q-learning, policy gradients)
    - Cloud LLM strategic guidance
    - MoE expert consensus
    - Consciousness-guided learning (Δ𝒞)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory: CloudRLMemory,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """Initialize cloud RL agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = memory
        
        # RL hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-network (simple numpy implementation for now)
        # In production, use PyTorch or TensorFlow
        self.q_network = np.random.randn(state_dim, action_dim) * 0.01
        
        # Statistics
        self.training_steps = 0
        self.episodes = 0
        
        logger.info("Cloud RL Agent initialized")
    
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploit: best action from Q-network
            q_values = state @ self.q_network
            return int(np.argmax(q_values))
    
    async def select_action_with_llm_guidance(
        self,
        state: np.ndarray,
        state_description: str,
        available_actions: List[str],
        context: Dict[str, Any]
    ) -> Tuple[int, str]:
        """
        Select action with cloud LLM strategic guidance.
        
        Returns:
            (action_index, llm_reasoning)
        """
        # Get Q-values
        q_values = state @ self.q_network
        
        # Get LLM suggestion
        llm_suggestion = None
        if self.memory.hybrid_llm and np.random.random() < 0.3:  # 30% of time
            try:
                prompt = f"""Current situation in Skyrim:
{state_description}

Available actions:
{chr(10).join(f'{i+1}. {action}' for i, action in enumerate(available_actions))}

Q-values suggest: {available_actions[np.argmax(q_values)]}

What action would you recommend and why? Consider both immediate tactics and long-term strategy.
Format: ACTION: <number>
REASONING: <text>"""

                response = await self.memory.hybrid_llm.generate_reasoning(
                    prompt=prompt,
                    system_prompt="You are a Skyrim strategy expert. Provide tactical advice.",
                    temperature=0.5,
                    max_tokens=256
                )
                
                # Parse LLM suggestion
                if "ACTION:" in response:
                    try:
                        action_line = [l for l in response.split('\n') if 'ACTION:' in l][0]
                        suggested_idx = int(action_line.split('ACTION:')[1].strip().split()[0]) - 1
                        
                        if 0 <= suggested_idx < len(available_actions):
                            llm_suggestion = suggested_idx
                    except:
                        pass
                
                logger.debug(f"LLM suggested action: {llm_suggestion}")
                
            except Exception as e:
                logger.error(f"LLM guidance failed: {e}")
        
        # Combine Q-learning and LLM guidance
        if llm_suggestion is not None and np.random.random() < 0.7:  # Trust LLM 70% of time
            action_idx = llm_suggestion
        else:
            action_idx = self.select_action(state, explore=True)
        
        return action_idx, llm_suggestion
    
    async def train_step(self, batch_size: int = 32):
        """Perform one training step using experience replay."""
        if len(self.memory.experiences) < batch_size:
            return
        
        # Sample batch (prioritized)
        batch = self.memory.sample_prioritized_batch(batch_size)
        
        # Simple Q-learning update (placeholder)
        # In production, use proper neural network training
        for exp in batch:
            # Compute target
            if exp.done:
                target = exp.reward + exp.llm_reward_adjustment
            else:
                next_q = np.max(exp.next_state_vector @ self.q_network)
                target = exp.reward + exp.llm_reward_adjustment + self.gamma * next_q
            
            # Update Q-network (gradient descent step)
            current_q = exp.state_vector @ self.q_network
            action_idx = 0  # Would need to track actual action index
            
            # Simple update rule
            self.q_network += self.learning_rate * (target - current_q[action_idx]) * np.outer(exp.state_vector, np.eye(self.action_dim)[action_idx])
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_steps += 1
    
    def save(self, path: str):
        """Save agent state."""
        try:
            save_dict = {
                'q_network': self.q_network,
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
                'episodes': self.episodes,
            }
            
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f)
            
            logger.info(f"Agent saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save agent: {e}")
    
    def load(self, path: str):
        """Load agent state."""
        try:
            if Path(path).exists():
                with open(path, 'rb') as f:
                    save_dict = pickle.load(f)
                
                self.q_network = save_dict['q_network']
                self.epsilon = save_dict['epsilon']
                self.training_steps = save_dict['training_steps']
                self.episodes = save_dict['episodes']
                
                logger.info(f"Agent loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
