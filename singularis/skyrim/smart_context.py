"""
Smart Context Management System

Optimizes LLM prompts by:
1. Selecting only relevant information based on task type
2. Prioritizing recent and important context
3. Compressing repetitive information
4. Adapting context size to model capabilities
5. Caching frequently used context

This reduces token usage, improves response speed, and enhances quality.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time
import hashlib

from loguru import logger


@dataclass
class ContextPriority:
    """Defines priority levels for including different pieces of context in a prompt.

    This class provides a standardized way to rank the importance of information,
    ensuring that the most critical data is always included within the token budget,
    while less important data is included only if space allows.
    """
    CRITICAL = 1.0   # Must include (current state, immediate danger)
    HIGH = 0.8       # Very important (recent actions, goals)
    MEDIUM = 0.5     # Useful (Q-values, scene info)
    LOW = 0.3        # Background (stats, general info)
    MINIMAL = 0.1    # Only if space allows


@dataclass
class ContextTemplate:
    """A template that specifies context priorities for a specific type of task.

    Each attribute corresponds to a category of information, and its value is a
    priority score from `ContextPriority`. This allows the `SmartContextManager`
    to tailor the context for different cognitive functions like vision analysis,
    planning, or strategic thinking.
    """
    vision_analysis: float = ContextPriority.HIGH
    world_modeling: float = ContextPriority.HIGH
    action_planning: float = ContextPriority.CRITICAL
    strategic_thinking: float = ContextPriority.MEDIUM
    ethical_reasoning: float = ContextPriority.MEDIUM


class SmartContextManager:
    """Manages and optimizes the context provided to Large Language Model (LLM) calls.

    This class is responsible for constructing a concise yet comprehensive context
    string that fits within a specified token budget. It uses task-specific templates,
    relevance scoring, caching, and adaptive compression to ensure the LLM has the
    most relevant information to perform its task effectively, reducing token usage
    and improving response quality.
    """
    
    def __init__(
        self,
        max_tokens_per_call: int = 2000,
        cache_size: int = 50,
        enable_compression: bool = True
    ):
        """Initializes the SmartContextManager.

        Args:
            max_tokens_per_call: The maximum number of tokens allowed for the
                                 context in a single LLM call.
            cache_size: The number of generated contexts to cache for reuse.
            enable_compression: A boolean to enable or disable context compression.
        """
        self.max_tokens = max_tokens_per_call
        self.cache_size = cache_size
        self.enable_compression = enable_compression
        
        # Context cache (hash -> content)
        self.context_cache: Dict[str, Tuple[str, float]] = {}  # hash -> (content, timestamp)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Recent context history
        self.recent_states = deque(maxlen=10)
        self.recent_actions = deque(maxlen=20)
        self.recent_outcomes = deque(maxlen=10)
        
        # Context templates for different tasks
        self.templates = {
            'vision': ContextTemplate(
                vision_analysis=1.0,
                world_modeling=0.3,
                action_planning=0.5,
                strategic_thinking=0.2,
                ethical_reasoning=0.1
            ),
            'reasoning': ContextTemplate(
                vision_analysis=0.5,
                world_modeling=0.8,
                action_planning=1.0,
                strategic_thinking=0.7,
                ethical_reasoning=0.6
            ),
            'world_model': ContextTemplate(
                vision_analysis=0.4,
                world_modeling=1.0,
                action_planning=0.6,
                strategic_thinking=0.9,
                ethical_reasoning=0.7
            ),
            'action': ContextTemplate(
                vision_analysis=0.7,
                world_modeling=0.4,
                action_planning=1.0,
                strategic_thinking=0.5,
                ethical_reasoning=0.3
            )
        }
        
        logger.info("[SMART-CONTEXT] Context manager initialized")
    
    def build_context(
        self,
        task_type: str,
        perception: Dict[str, Any],
        state_dict: Dict[str, Any],
        q_values: Optional[Dict[str, float]] = None,
        available_actions: Optional[List[str]] = None,
        curriculum_knowledge: Optional[str] = None,
        memory_context: Optional[str] = None
    ) -> str:
        """Constructs an optimized context string for a given task.

        This method dynamically assembles the most relevant pieces of information—such
        as current state, recent history, available actions, and knowledge from RAG
        systems—into a single string, respecting the token budget. It uses a
        task-specific template to prioritize what to include.

        Args:
            task_type: The type of task (e.g., 'vision', 'reasoning', 'action').
            perception: Data from the perception system, including scene analysis.
            state_dict: The current state of the agent and the environment.
            q_values: A dictionary of Q-values for available actions.
            available_actions: A list of actions the agent can currently take.
            curriculum_knowledge: Knowledge retrieved from the curriculum RAG.
            memory_context: Context retrieved from the memory RAG.

        Returns:
            A formatted and optimized string to be used as context in an LLM prompt.
        """
        template = self.templates.get(task_type, self.templates['reasoning'])
        
        # Build context sections with priorities
        sections = []
        token_budget = self.max_tokens
        
        # 1. Critical: Current state (always include)
        current_state = self._format_current_state(state_dict, perception)
        sections.append(('CURRENT STATE', current_state, ContextPriority.CRITICAL))
        token_budget -= self._estimate_tokens(current_state)
        
        # 2. High: Available actions and goals
        if available_actions and template.action_planning >= 0.5:
            actions_text = self._format_actions(available_actions, q_values)
            sections.append(('AVAILABLE ACTIONS', actions_text, ContextPriority.HIGH))
            token_budget -= self._estimate_tokens(actions_text)
        
        # 3. Medium: Recent history (compressed)
        if template.strategic_thinking >= 0.5:
            history_text = self._format_recent_history()
            if history_text and token_budget > 200:
                sections.append(('RECENT HISTORY', history_text, ContextPriority.MEDIUM))
                token_budget -= self._estimate_tokens(history_text)
        
        # 4. Medium: Scene and environmental context
        if template.vision_analysis >= 0.5:
            scene_text = self._format_scene_context(perception)
            if scene_text and token_budget > 150:
                sections.append(('ENVIRONMENT', scene_text, ContextPriority.MEDIUM))
                token_budget -= self._estimate_tokens(scene_text)
        
        # 5. Low: Academic knowledge (if space allows)
        if curriculum_knowledge and template.ethical_reasoning >= 0.5 and token_budget > 300:
            knowledge_compressed = self._compress_knowledge(curriculum_knowledge, max_length=400)
            sections.append(('ACADEMIC KNOWLEDGE', knowledge_compressed, ContextPriority.LOW))
            token_budget -= self._estimate_tokens(knowledge_compressed)
        
        # 6. Low: Memory context (if space allows)
        if memory_context and token_budget > 200:
            memory_compressed = self._compress_text(memory_context, max_length=300)
            sections.append(('RELEVANT MEMORIES', memory_compressed, ContextPriority.LOW))
            token_budget -= self._estimate_tokens(memory_compressed)
        
        # Build final context
        context_parts = []
        for title, content, priority in sections:
            if content:
                context_parts.append(f"[{title}]")
                context_parts.append(content)
                context_parts.append("")  # Blank line
        
        final_context = "\n".join(context_parts)
        
        # Cache the context
        self._cache_context(task_type, final_context)
        
        return final_context
    
    def build_prompt_with_context(
        self,
        base_prompt: str,
        task_type: str,
        **context_kwargs
    ) -> str:
        """Combines a base prompt with a dynamically generated smart context.

        Args:
            base_prompt: The core instruction or question for the LLM.
            task_type: The type of task, used to build the appropriate context.
            **context_kwargs: Keyword arguments passed to the `build_context` method.

        Returns:
            A complete LLM prompt string, including both context and the task.
        """
        context = self.build_context(task_type, **context_kwargs)
        
        # Combine with base prompt
        full_prompt = f"{context}\n\n[TASK]\n{base_prompt}"
        
        return full_prompt
    
    def update_history(
        self,
        state: Dict[str, Any],
        action: str,
        outcome: Dict[str, Any]
    ):
        """Updates the manager's history with the latest state-action-outcome tuple.

        This information is used to provide recent context for future LLM calls,
        helping the model understand the immediate sequence of events.

        Args:
            state: The state before the action was taken.
            action: The action that was performed.
            outcome: The resulting outcome or state change.
        """
        self.recent_states.append({
            'timestamp': time.time(),
            'state': state
        })
        self.recent_actions.append({
            'timestamp': time.time(),
            'action': action
        })
        self.recent_outcomes.append({
            'timestamp': time.time(),
            'outcome': outcome
        })
    
    def _format_current_state(
        self,
        state_dict: Dict[str, Any],
        perception: Dict[str, Any]
    ) -> str:
        """Formats the current state into a concise string."""
        parts = []
        
        # Critical stats
        health = state_dict.get('health', 100)
        stamina = state_dict.get('stamina', 100)
        magicka = state_dict.get('magicka', 100)
        parts.append(f"Health: {health}% | Stamina: {stamina}% | Magicka: {magicka}%")
        
        # Combat status
        in_combat = state_dict.get('in_combat', False)
        enemies = state_dict.get('enemies_nearby', 0)
        if in_combat or enemies > 0:
            parts.append(f"⚠️ COMBAT: {enemies} enemies nearby")
        
        # Scene
        scene = perception.get('scene_type', 'unknown')
        parts.append(f"Scene: {scene}")
        
        # Location (if available)
        location = state_dict.get('location', '')
        if location:
            parts.append(f"Location: {location}")
        
        return "\n".join(parts)
    
    def _format_actions(
        self,
        available_actions: List[str],
        q_values: Optional[Dict[str, float]] = None
    ) -> str:
        """Formats the list of available actions, optionally with their Q-values."""
        if not available_actions:
            return "No actions available"
        
        if q_values:
            # Sort by Q-value
            sorted_actions = sorted(
                available_actions,
                key=lambda a: q_values.get(a, 0.0),
                reverse=True
            )
            lines = []
            for action in sorted_actions[:10]:  # Top 10
                q_val = q_values.get(action, 0.0)
                confidence = "HIGH" if q_val > 0.5 else "MEDIUM" if q_val > 0.2 else "LOW"
                lines.append(f"  • {action} (Q={q_val:.2f}, {confidence})")
            return "\n".join(lines)
        else:
            return ", ".join(available_actions[:15])
    
    def _format_recent_history(self) -> str:
        """Formats the recent action history into a concise string."""
        if not self.recent_actions:
            return ""
        
        # Get last 5 actions
        recent = list(self.recent_actions)[-5:]
        lines = ["Last actions:"]
        for i, action_data in enumerate(recent, 1):
            action = action_data.get('action', 'unknown')
            lines.append(f"  {i}. {action}")
        
        return "\n".join(lines)
    
    def _format_scene_context(self, perception: Dict[str, Any]) -> str:
        """Formats the environmental and scene context from perception data."""
        parts = []
        
        scene_type = perception.get('scene_type', 'unknown')
        parts.append(f"Type: {scene_type}")
        
        # Visual description (if available)
        if 'visual_description' in perception:
            desc = perception['visual_description'][:200]  # Truncate
            parts.append(f"Visual: {desc}")
        
        return "\n".join(parts) if parts else ""
    
    def _compress_knowledge(self, knowledge: str, max_length: int = 400) -> str:
        """Compresses academic knowledge by extracting key sentences."""
        if len(knowledge) <= max_length:
            return knowledge
        
        # Extract key sentences (simple heuristic: sentences with important words)
        sentences = knowledge.split('.')
        important_words = {'must', 'should', 'important', 'critical', 'key', 'principle', 'because', 'therefore'}
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            score = sum(1 for word in important_words if word in sentence.lower())
            scored_sentences.append((score, sentence))
        
        # Take top sentences
        scored_sentences.sort(reverse=True)
        selected = [s for _, s in scored_sentences[:3]]
        
        compressed = '. '.join(selected)
        if len(compressed) > max_length:
            compressed = compressed[:max_length-3] + "..."
        
        return compressed
    
    def _compress_text(self, text: str, max_length: int = 300) -> str:
        """Compresses text by simple truncation."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimates the token count of a string using a simple heuristic."""
        # Simple heuristic: ~4 chars per token
        return len(text) // 4
    
    def _cache_context(self, task_type: str, context: str):
        """Caches a generated context string for potential reuse."""
        cache_key = hashlib.md5(f"{task_type}:{context[:100]}".encode()).hexdigest()
        
        # Prune old cache entries
        if len(self.context_cache) >= self.cache_size:
            # Remove oldest
            oldest_key = min(self.context_cache.keys(), 
                           key=lambda k: self.context_cache[k][1])
            del self.context_cache[oldest_key]
        
        self.context_cache[cache_key] = (context, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the context manager's performance.

        Returns:
            A dictionary containing cache statistics (hits, misses, hit rate)
            and the size of the history deques.
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            'cache_size': len(self.context_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'recent_states': len(self.recent_states),
            'recent_actions': len(self.recent_actions),
            'recent_outcomes': len(self.recent_outcomes)
        }
