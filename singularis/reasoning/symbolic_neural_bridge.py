"""
Symbolic-Neural Bridge: Hybrid Reasoning System

Integrates symbolic logic with neural LLM reasoning:
- Symbolic rules gate expensive LLM calls
- Memory-based reasoning enhances LLM capabilities
- Neural-symbolic feedback loop for continuous improvement
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ReasoningMode(Enum):
    """Reasoning mode selection."""
    SYMBOLIC_ONLY = "symbolic_only"  # Fast, rule-based
    NEURAL_ONLY = "neural_only"      # LLM-based
    HYBRID = "hybrid"                # Both, symbolic gates neural
    MEMORY_GUIDED = "memory_guided"  # Memory retrieval + reasoning


@dataclass
class ReasoningDecision:
    """Decision about how to reason."""
    mode: ReasoningMode
    confidence: float
    reasoning: str
    should_invoke_llm: bool
    symbolic_result: Optional[Any] = None
    memory_context: Optional[List[Dict]] = None
    estimated_cost: float = 0.0  # API cost estimate


class SymbolicGate:
    """
    Symbolic logic gate that decides when to invoke LLMs.
    
    Uses fast symbolic rules to:
    - Handle simple cases without LLM
    - Detect when LLM is needed
    - Provide context for LLM calls
    """
    
    def __init__(self, rule_engine, memory_system):
        """
        Initialize symbolic gate.
        self.omega = omega
        
        Args:
            rule_engine: Symbolic rule engine
            memory_system: Hierarchical memory system
        """
        self.rule_engine = rule_engine
        self.memory_system = memory_system
        
        # Gate statistics
        self.stats = {
            'total_decisions': 0,
            'symbolic_only': 0,
            'neural_invoked': 0,
            'memory_guided': 0,
            'llm_calls_saved': 0,
            'total_cost_saved': 0.0,
        }
    
    def should_invoke_llm(
        self,
        query: str,
        context: Dict[str, Any],
        complexity_threshold: float = 0.7,
    ) -> ReasoningDecision:
        """
        Decide whether to invoke LLM based on symbolic analysis.
        
        Args:
            query: Reasoning query
            context: Current context
            complexity_threshold: Threshold for LLM invocation
            
        Returns:
            ReasoningDecision with mode and context
        """
        self.stats['total_decisions'] += 1
        
        # 1. Check if symbolic rules can handle this
        symbolic_result = self._try_symbolic_reasoning(query, context)
        
        if symbolic_result['can_handle']:
            # Symbolic rules are sufficient
            self.stats['symbolic_only'] += 1
            self.stats['llm_calls_saved'] += 1
            self.stats['total_cost_saved'] += 0.01  # Estimated LLM cost
            
            return ReasoningDecision(
                mode=ReasoningMode.SYMBOLIC_ONLY,
                confidence=symbolic_result['confidence'],
                reasoning=f"Symbolic rule: {symbolic_result['rule_name']}",
                should_invoke_llm=False,
                symbolic_result=symbolic_result['result'],
                estimated_cost=0.0,
            )
        
        # 2. Check memory for similar cases
        memory_context = self._retrieve_relevant_memories(query, context)
        
        if memory_context and len(memory_context) >= 3:
            # Strong memory guidance available
            memory_confidence = np.mean([m['confidence'] for m in memory_context])
            
            if memory_confidence > 0.8:
                # Memory is highly confident - use memory-guided reasoning
                self.stats['memory_guided'] += 1
                self.stats['llm_calls_saved'] += 1
                self.stats['total_cost_saved'] += 0.01
                
                return ReasoningDecision(
                    mode=ReasoningMode.MEMORY_GUIDED,
                    confidence=memory_confidence,
                    reasoning=f"Memory-guided: {len(memory_context)} similar cases",
                    should_invoke_llm=False,
                    memory_context=memory_context,
                    estimated_cost=0.0,
                )
        
        # 3. Assess query complexity
        complexity = self._assess_complexity(query, context)
        
        if complexity < complexity_threshold:
            # Simple enough for hybrid approach
            return ReasoningDecision(
                mode=ReasoningMode.HYBRID,
                confidence=0.6,
                reasoning=f"Hybrid: complexity={complexity:.2f}, using symbolic + light LLM",
                should_invoke_llm=True,
                symbolic_result=symbolic_result.get('partial_result'),
                memory_context=memory_context,
                estimated_cost=0.005,  # Cheaper LLM call with context
            )
        
        # 4. Complex query - full LLM reasoning needed
        self.stats['neural_invoked'] += 1
        
        return ReasoningDecision(
            mode=ReasoningMode.NEURAL_ONLY,
            confidence=0.5,
            reasoning=f"Complex query: complexity={complexity:.2f}, full LLM needed",
            should_invoke_llm=True,
            symbolic_result=None,
            memory_context=memory_context,
            estimated_cost=0.01,
        )
    
    def _try_symbolic_reasoning(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Try to answer query with symbolic rules."""
        # Check if rule engine can handle this
        rule_results = self.rule_engine.evaluate(context)
        
        # Look for direct answers in rules
        for rule_name, rule_data in rule_results.get('facts', {}).items():
            if self._query_matches_rule(query, rule_name, rule_data):
                return {
                    'can_handle': True,
                    'confidence': rule_data.get('confidence', 0.9),
                    'rule_name': rule_name,
                    'result': rule_data.get('value'),
                }
        
        # Check recommendations
        recommendations = rule_results.get('recommendations', [])
        if recommendations and self._query_about_action(query):
            return {
                'can_handle': True,
                'confidence': 0.85,
                'rule_name': 'action_recommendation',
                'result': recommendations[0] if recommendations else None,
            }
        
        # Cannot handle symbolically
        return {
            'can_handle': False,
            'confidence': 0.0,
            'partial_result': rule_results,  # Provide as context
        }
    
    def _retrieve_relevant_memories(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for this query."""
        if not self.memory_system:
            return []
        
        # Search semantic memory for similar situations
        try:
            memories = self.memory_system.retrieve_semantic(
                query=query,
                context=context,
                top_k=5,
            )
            return memories
        except Exception as e:
            print(f"[SYMBOLIC-GATE] Memory retrieval error: {e}")
            return []
    
    def _assess_complexity(self, query: str, context: Dict[str, Any]) -> float:
        """Assess query complexity (0-1)."""
        complexity = 0.0
        
        # Length-based complexity
        if len(query) > 200:
            complexity += 0.2
        elif len(query) > 100:
            complexity += 0.1
        
        # Multi-step reasoning indicators
        multi_step_keywords = ['then', 'after', 'before', 'because', 'if', 'while']
        if any(kw in query.lower() for kw in multi_step_keywords):
            complexity += 0.2
        
        # Abstract concepts
        abstract_keywords = ['why', 'how', 'explain', 'understand', 'reason']
        if any(kw in query.lower() for kw in abstract_keywords):
            complexity += 0.3
        
        # Context complexity
        if len(context) > 10:
            complexity += 0.1
        
        # Uncertainty indicators
        if context.get('uncertainty', 0.0) > 0.5:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _query_matches_rule(
        self,
        query: str,
        rule_name: str,
        rule_data: Dict
    ) -> bool:
        """Check if query matches a rule."""
        query_lower = query.lower()
        rule_lower = rule_name.lower()
        
        # Simple keyword matching
        return any(word in query_lower for word in rule_lower.split('_'))
    
    def _query_about_action(self, query: str) -> bool:
        """Check if query is about action selection."""
        action_keywords = ['should', 'do', 'action', 'next', 'move', 'attack']
        return any(kw in query.lower() for kw in action_keywords)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gating statistics."""
        total = self.stats['total_decisions']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'symbolic_rate': self.stats['symbolic_only'] / total,
            'neural_rate': self.stats['neural_invoked'] / total,
            'memory_rate': self.stats['memory_guided'] / total,
            'llm_reduction': self.stats['llm_calls_saved'] / total,
        }


class MemoryGuidedReasoning:
    """
    Memory-guided reasoning that enhances LLM calls.
    
    Uses retrieved memories to:
    - Provide examples to LLM
    - Guide reasoning direction
    - Validate LLM outputs
    """
    
    def __init__(self, memory_system):
        """Initialize memory-guided reasoning."""
        self.memory_system = memory_system
    
    async def reason_with_memory(
        self,
        query: str,
        context: Dict[str, Any],
        memories: List[Dict[str, Any]],
        llm_client,
    ) -> Dict[str, Any]:
        """
        Perform reasoning guided by memories.
        
        Args:
            query: Reasoning query
            context: Current context
            memories: Retrieved relevant memories
            llm_client: LLM client for reasoning
            
        Returns:
            Reasoning result with confidence
        """
        # Build memory-enhanced prompt
        prompt = self._build_memory_prompt(query, context, memories)
        
        # Call LLM with memory context
        try:
            response = await llm_client.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temp for more consistent reasoning
                max_tokens=500,
            )
            
            # Extract reasoning
            reasoning = response.get('text', '')
            
            # Validate against memories
            validation = self._validate_with_memories(reasoning, memories)
            
            return {
                'reasoning': reasoning,
                'confidence': validation['confidence'],
                'memory_support': validation['support_count'],
                'contradictions': validation['contradictions'],
            }
            
        except Exception as e:
            print(f"[MEMORY-REASONING] LLM error: {e}")
            # Fallback to memory-only reasoning
            return self._memory_only_reasoning(memories)
    
    def _build_memory_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> str:
        """Build prompt enhanced with memory examples."""
        prompt = f"Query: {query}\n\n"
        prompt += f"Current Context:\n{self._format_context(context)}\n\n"
        
        if memories:
            prompt += "Relevant Past Experiences:\n"
            for i, memory in enumerate(memories[:3], 1):
                prompt += f"{i}. {memory.get('description', 'N/A')}\n"
                prompt += f"   Outcome: {memory.get('outcome', 'N/A')}\n"
                prompt += f"   Confidence: {memory.get('confidence', 0.0):.2f}\n\n"
        
        prompt += "Based on the current context and past experiences, provide your reasoning:\n"
        
        return prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        lines = []
        for key, value in context.items():
            if isinstance(value, (int, float, str, bool)):
                lines.append(f"- {key}: {value}")
        return '\n'.join(lines[:10])  # Limit to 10 items
    
    def _validate_with_memories(
        self,
        reasoning: str,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate reasoning against memories."""
        support_count = 0
        contradictions = []
        
        for memory in memories:
            memory_outcome = memory.get('outcome', '')
            
            # Simple validation: check if reasoning aligns with memory outcomes
            if memory_outcome and memory_outcome.lower() in reasoning.lower():
                support_count += 1
            elif memory_outcome and self._contradicts(reasoning, memory_outcome):
                contradictions.append(memory.get('description', 'Unknown'))
        
        # Confidence based on support
        confidence = min(0.5 + (support_count * 0.15), 0.95)
        
        return {
            'confidence': confidence,
            'support_count': support_count,
            'contradictions': contradictions,
        }
    
    def _contradicts(self, reasoning: str, outcome: str) -> bool:
        """Check if reasoning contradicts outcome."""
        # Simple contradiction detection
        negation_words = ['not', 'never', 'avoid', 'don\'t']
        
        reasoning_lower = reasoning.lower()
        outcome_lower = outcome.lower()
        
        # If outcome suggests action but reasoning suggests opposite
        if any(neg in reasoning_lower for neg in negation_words):
            if outcome_lower in reasoning_lower:
                return True
        
        return False
    
    def _memory_only_reasoning(
        self,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback: reason using only memories."""
        if not memories:
            return {
                'reasoning': "No memories available for reasoning",
                'confidence': 0.1,
                'memory_support': 0,
                'contradictions': [],
            }
        
        # Aggregate memory outcomes
        outcomes = [m.get('outcome', '') for m in memories if m.get('outcome')]
        
        if outcomes:
            # Use most confident memory
            best_memory = max(memories, key=lambda m: m.get('confidence', 0.0))
            
            return {
                'reasoning': f"Based on past experience: {best_memory.get('description', 'N/A')}",
                'confidence': best_memory.get('confidence', 0.5),
                'memory_support': len(outcomes),
                'contradictions': [],
            }
        
        return {
            'reasoning': "Insufficient memory data",
            'confidence': 0.2,
            'memory_support': 0,
            'contradictions': [],
        }


class SymbolicNeuralBridge:
    """
    Main bridge between symbolic and neural reasoning.
    
    Orchestrates:
    - Symbolic gating of LLM calls
    - Memory-guided reasoning
    - Hybrid symbolic-neural inference
    - Feedback loop for improvement
    """
    
    def __init__(
        self,
        rule_engine,
        memory_system,
        moe_orchestrator,
        enable_gating: bool = True,
        enable_memory_guidance: bool = True,
        omega=None,
    ):
        """
        Initialize symbolic-neural bridge.
        
        Args:
            rule_engine: Symbolic rule engine
            memory_system: Hierarchical memory system
            moe_orchestrator: MoE LLM orchestrator
            enable_gating: Enable symbolic gating
            enable_memory_guidance: Enable memory-guided reasoning
        """
        self.rule_engine = rule_engine
        self.memory_system = memory_system
        self.moe = moe_orchestrator
        
        self.enable_gating = enable_gating
        self.enable_memory_guidance = enable_memory_guidance
        
        # Components
        self.symbolic_gate = SymbolicGate(rule_engine, memory_system)
        self.memory_reasoning = MemoryGuidedReasoning(memory_system)
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'symbolic_resolved': 0,
            'memory_resolved': 0,
            'llm_invoked': 0,
            'hybrid_used': 0,
            'avg_confidence': 0.0,
        }
    
    async def reason(
        self,
        query: str,
        context: Dict[str, Any],
        require_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform reasoning using optimal combination of symbolic/neural.
        
        Args:
            query: Reasoning query
            context: Current context
            require_llm: Force LLM usage
            
        Returns:
            Reasoning result with confidence and explanation
        """
        self.stats['total_queries'] += 1
        start_time = time.time()
        
        # 1. Symbolic gating decision
        if self.enable_gating and not require_llm:
            decision = self.symbolic_gate.should_invoke_llm(query, context)
        else:
            # Forced LLM mode
            decision = ReasoningDecision(
                mode=ReasoningMode.NEURAL_ONLY,
                confidence=0.5,
                reasoning="LLM required",
                should_invoke_llm=True,
            )
        
        # OMEGA: record gating event if available
        try:
            if hasattr(self, 'omega') and self.omega is not None:
                self.omega.record_gating_event({
                    'mode': decision.mode.value if hasattr(decision.mode, 'value') else str(decision.mode),
                    'confidence': decision.confidence,
                    'should_invoke_llm': decision.should_invoke_llm
                }, context)
        except Exception:
            pass

        # 2. Execute based on decision
        if decision.mode == ReasoningMode.SYMBOLIC_ONLY:
            # Pure symbolic reasoning
            self.stats['symbolic_resolved'] += 1
            result = {
                'answer': decision.symbolic_result,
                'confidence': decision.confidence,
                'mode': 'symbolic',
                'reasoning': decision.reasoning,
                'cost': 0.0,
            }
        
        elif decision.mode == ReasoningMode.MEMORY_GUIDED:
            # Memory-guided reasoning (no LLM)
            self.stats['memory_resolved'] += 1
            
            if self.enable_memory_guidance and decision.memory_context:
                # Use memory-only reasoning
                memory_result = self.memory_reasoning._memory_only_reasoning(
                    decision.memory_context
                )
                result = {
                    'answer': memory_result['reasoning'],
                    'confidence': memory_result['confidence'],
                    'mode': 'memory',
                    'reasoning': decision.reasoning,
                    'memory_support': memory_result['memory_support'],
                    'cost': 0.0,
                }
            else:
                # Fallback to symbolic
                result = {
                    'answer': None,
                    'confidence': 0.3,
                    'mode': 'memory_fallback',
                    'reasoning': "Memory guidance unavailable",
                    'cost': 0.0,
                }
        
        elif decision.mode == ReasoningMode.HYBRID:
            # Hybrid: symbolic context + light LLM
            self.stats['hybrid_used'] += 1
            self.stats['llm_invoked'] += 1
            
            # Build hybrid prompt with symbolic context
            hybrid_prompt = self._build_hybrid_prompt(
                query, context, decision.symbolic_result
            )
            
            # Light LLM call
            # OMEGA: record MoE hybrid query
            try:
                if hasattr(self, 'omega') and self.omega is not None:
                    self.omega.record_moe_query('hybrid')
            except Exception:
                pass
            llm_result = await self._invoke_llm(hybrid_prompt, decision.memory_context)
            
            result = {
                'answer': llm_result['answer'],
                'confidence': llm_result['confidence'],
                'mode': 'hybrid',
                'reasoning': f"Symbolic + LLM: {decision.reasoning}",
                'symbolic_context': decision.symbolic_result,
                'cost': decision.estimated_cost,
            }
        
        else:  # NEURAL_ONLY
            # Full LLM reasoning
            self.stats['llm_invoked'] += 1
            
            if self.enable_memory_guidance and decision.memory_context:
                # Memory-guided LLM
                # OMEGA: record MoE reasoning query
                try:
                    if hasattr(self, 'omega') and self.omega is not None:
                        self.omega.record_moe_query('reasoning')
                except Exception:
                    pass
                llm_result = await self.memory_reasoning.reason_with_memory(
                    query, context, decision.memory_context, self.moe
                )
                result = {
                    'answer': llm_result['reasoning'],
                    'confidence': llm_result['confidence'],
                    'mode': 'neural_memory_guided',
                    'reasoning': decision.reasoning,
                    'memory_support': llm_result['memory_support'],
                    'cost': decision.estimated_cost,
                }
            else:
                # Pure LLM
                try:
                    if hasattr(self, 'omega') and self.omega is not None:
                        self.omega.record_moe_query('reasoning')
                except Exception:
                    pass
                llm_result = await self._invoke_llm(query, None)
                result = {
                    'answer': llm_result['answer'],
                    'confidence': llm_result['confidence'],
                    'mode': 'neural',
                    'reasoning': decision.reasoning,
                    'cost': decision.estimated_cost,
                }
        
        # 3. Store result in memory for future guidance
        if result['confidence'] > 0.6:
            self._store_reasoning_memory(query, context, result)
        
        # 4. Update statistics
        duration = time.time() - start_time
        result['duration'] = duration
        
        self._update_stats(result)
        
        return result
    
    def _build_hybrid_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        symbolic_result: Any
    ) -> str:
        """Build prompt for hybrid reasoning."""
        prompt = f"Query: {query}\n\n"
        
        if symbolic_result:
            prompt += f"Symbolic Analysis:\n{symbolic_result}\n\n"
        
        prompt += f"Context: {context.get('scene', 'unknown')}\n"
        prompt += f"Health: {context.get('health', 100)}\n"
        prompt += f"In Combat: {context.get('in_combat', False)}\n\n"
        
        prompt += "Provide concise reasoning (1-2 sentences):\n"
        
        return prompt
    
    async def _invoke_llm(
        self,
        prompt: str,
        memory_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Invoke LLM (MoE) for reasoning."""
        try:
            # Use MoE for consensus
            response = await self.moe.query_experts(
                prompt=prompt,
                num_experts=3,  # Lighter MoE call
                require_consensus=False,
            )
            
            return {
                'answer': response.get('consensus', ''),
                'confidence': response.get('confidence', 0.5),
            }
            
        except Exception as e:
            print(f"[SYMBOLIC-NEURAL] LLM error: {e}")
            return {
                'answer': "LLM unavailable",
                'confidence': 0.1,
            }
    
    def _store_reasoning_memory(
        self,
        query: str,
        context: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Store reasoning result in memory."""
        if not self.memory_system:
            return
        
        try:
            self.memory_system.store_episodic(
                description=f"Reasoning: {query[:100]}",
                outcome=result['answer'],
                context=context,
                confidence=result['confidence'],
            )
        except Exception as e:
            print(f"[SYMBOLIC-NEURAL] Memory storage error: {e}")
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update running statistics."""
        # Update average confidence
        total = self.stats['total_queries']
        old_avg = self.stats['avg_confidence']
        new_conf = result['confidence']
        
        self.stats['avg_confidence'] = (old_avg * (total - 1) + new_conf) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        total = self.stats['total_queries']
        if total == 0:
            return self.stats
        
        gate_stats = self.symbolic_gate.get_stats()
        
        return {
            **self.stats,
            'symbolic_rate': self.stats['symbolic_resolved'] / total,
            'memory_rate': self.stats['memory_resolved'] / total,
            'llm_rate': self.stats['llm_invoked'] / total,
            'hybrid_rate': self.stats['hybrid_used'] / total,
            'gating_stats': gate_stats,
            'cost_savings': gate_stats.get('total_cost_saved', 0.0),
        }
