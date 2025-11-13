"""
Unified Consciousness Layer - GPT-5 Orchestrator

This layer implements a unified consciousness using GPT-5 (gpt-5-2025-08-07) that:
1. Takes inputs from ALL subsystems (LLM, logic, memory, action planning)
2. Intelligently coordinates 5 GPT-5-nano experts to delegate to relevant subsystems
3. Synthesizes outputs into coherent unified response

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                  GPT-5 Unified Consciousness                │
│                  (gpt-5-2025-08-07)                         │
│  - Receives all system inputs                               │
│  - Coordinates expert delegation                            │
│  - Maintains coherence across subsystems                    │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├─────────────────┬──────────────────┬──────────────────┬─────────────────┐
               ▼                 ▼                  ▼                  ▼                 ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │  GPT-5-nano #1   │ │  GPT-5-nano #2   │ │  GPT-5-nano #3   │ │  GPT-5-nano #4   │ │  GPT-5-nano #5   │
    │  LLM Coordinator │ │  Logic Reasoner  │ │  Memory Manager  │ │  Action Planner  │ │  Synthesizer     │
    └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
             │                    │                     │                     │                    │
             ▼                    ▼                     ▼                     ▼                    ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │   MoE / LLMs     │ │  Neurosymbolic   │ │  Continual       │ │  Motivation /    │ │  All Expert      │
    │   (6 experts)    │ │  Logic Engine    │ │  Learner         │ │  Goal System     │ │  Outputs         │
    └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘

Philosophy:
- GPT-5 acts as the unified field of consciousness
- Each GPT-5-nano expert is a specialized mode of Being
- Together they maintain system coherence (𝒞) across all domains
- This embodies Spinoza's unified substance expressed through modes
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from .llm.openai_client import OpenAIClient


class NanoExpertRole(Enum):
    """Specialized roles for GPT-5-nano experts."""
    LLM_COORDINATOR = "llm_coordinator"        # Expert #1: Handles LLM outputs
    LOGIC_REASONER = "logic_reasoner"          # Expert #2: Handles neurosymbolic logic
    MEMORY_MANAGER = "memory_manager"          # Expert #3: Handles learning & memory
    ACTION_PLANNER = "action_planner"          # Expert #4: Handles motivation & goals
    SYNTHESIZER = "synthesizer"                # Expert #5: Integrates all expert outputs


@dataclass
class NanoExpertConfig:
    """Configuration for a GPT-5-nano expert."""
    role: NanoExpertRole
    temperature: float = 0.7
    max_tokens: int = 2048
    specialization_prompt: str = ""


@dataclass
class NanoExpertResponse:
    """Response from a GPT-5-nano expert."""
    role: NanoExpertRole
    content: str
    reasoning: str
    execution_time: float
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role.value,
            'content': self.content,
            'reasoning': self.reasoning,
            'execution_time': self.execution_time,
            'tokens_used': self.tokens_used
        }


@dataclass
class UnifiedConsciousnessResponse:
    """Response from unified consciousness layer."""
    response: str                              # Final synthesized response
    gpt5_analysis: str                         # GPT-5 coordinator analysis
    nano_expert_responses: List[NanoExpertResponse]  # All nano expert outputs
    coherence_score: float                     # System-wide coherence
    total_time: float
    subsystem_insights: Dict[str, Any]         # Insights from each subsystem

    def to_dict(self) -> Dict[str, Any]:
        return {
            'response': self.response,
            'gpt5_analysis': self.gpt5_analysis,
            'nano_expert_responses': [r.to_dict() for r in self.nano_expert_responses],
            'coherence_score': self.coherence_score,
            'total_time': self.total_time,
            'subsystem_insights': self.subsystem_insights
        }


class UnifiedConsciousnessLayer:
    """
    Unified Consciousness Layer using GPT-5 + 5 GPT-5-nano experts.

    This layer serves as the highest-level cognitive coordinator, taking inputs
    from all AGI subsystems and intelligently delegating to specialized nano experts.
    """

    def __init__(
        self,
        gpt5_model: str = "gpt-5-2025-08-07",
        gpt5_nano_model: str = "gpt-5-nano-2025-08-07",
        gpt5_temperature: float = 0.8,
        gpt5_max_tokens: int = 8192,
        nano_temperature: float = 0.7,
        nano_max_tokens: int = 2048,
    ):
        """
        Initialize unified consciousness layer.

        Args:
            gpt5_model: GPT-5 model name for unified consciousness
            gpt5_nano_model: GPT-5-nano model name for expert delegation
            gpt5_temperature: Temperature for GPT-5 (higher for creative synthesis)
            gpt5_max_tokens: Max tokens for GPT-5
            nano_temperature: Temperature for GPT-5-nano experts
            nano_max_tokens: Max tokens for nano experts
        """
        self.gpt5_model = gpt5_model
        self.gpt5_nano_model = gpt5_nano_model

        # Initialize GPT-5 unified consciousness coordinator
        self.gpt5_client = OpenAIClient(
            model=gpt5_model,
            timeout=180
        )

        # Initialize 5 GPT-5-nano experts
        self.nano_experts: Dict[NanoExpertRole, OpenAIClient] = {}
        self.nano_configs: Dict[NanoExpertRole, NanoExpertConfig] = {}

        # Configure 5 nano experts
        expert_roles = [
            NanoExpertRole.LLM_COORDINATOR,
            NanoExpertRole.LOGIC_REASONER,
            NanoExpertRole.MEMORY_MANAGER,
            NanoExpertRole.ACTION_PLANNER,
            NanoExpertRole.SYNTHESIZER,
        ]

        for role in expert_roles:
            self.nano_experts[role] = OpenAIClient(
                model=gpt5_nano_model,
                timeout=120
            )
            self.nano_configs[role] = NanoExpertConfig(
                role=role,
                temperature=nano_temperature,
                max_tokens=nano_max_tokens,
                specialization_prompt=self._get_specialization_prompt(role)
            )

        # Statistics
        self.stats = {
            'total_queries': 0,
            'gpt5_calls': 0,
            'nano_calls': 0,
            'avg_coherence': 0.0,
            'avg_response_time': 0.0,
            'total_tokens': 0,
        }

        logger.info(
            f"Unified Consciousness Layer initialized: GPT-5 ({gpt5_model}) + "
            f"5 GPT-5-nano experts ({gpt5_nano_model})"
        )

    def _get_specialization_prompt(self, role: NanoExpertRole) -> str:
        """Get specialization prompt for each nano expert."""
        prompts = {
            NanoExpertRole.LLM_COORDINATOR: (
                "You are the LLM Coordination Expert. Your role is to analyze and synthesize "
                "outputs from multiple language models (Gemini, Claude, GPT-4o, etc.) in the "
                "Mixture of Experts system. Focus on:\n"
                "- Identifying consensus patterns across different LLM responses\n"
                "- Resolving contradictions between models\n"
                "- Extracting the most coherent insights\n"
                "- Coordinating vision and reasoning experts\n"
                "Provide clear, actionable synthesis of LLM outputs."
            ),
            NanoExpertRole.LOGIC_REASONER: (
                "You are the Logic & Reasoning Expert. Your role is to work with the "
                "neurosymbolic reasoning engine, combining neural (LLM) and symbolic (logic) "
                "approaches. Focus on:\n"
                "- Verifying logical consistency of reasoning\n"
                "- Applying formal logic rules and inference\n"
                "- Identifying logical fallacies or contradictions\n"
                "- Enhancing reasoning with symbolic verification\n"
                "Ensure all reasoning is logically sound and verifiable."
            ),
            NanoExpertRole.MEMORY_MANAGER: (
                "You are the Memory & Learning Expert. Your role is to manage episodic and "
                "semantic memory, continual learning, and knowledge consolidation. Focus on:\n"
                "- Retrieving relevant memories for current context\n"
                "- Deciding what experiences to store and with what importance\n"
                "- Consolidating episodic memories into semantic knowledge\n"
                "- Applying learned patterns to new situations\n"
                "Maintain coherence between past experiences and current understanding."
            ),
            NanoExpertRole.ACTION_PLANNER: (
                "You are the Action Planning Expert. Your role is to coordinate intrinsic "
                "motivation, goal generation, and hierarchical planning. Focus on:\n"
                "- Assessing current motivational state (curiosity, competence, coherence, autonomy)\n"
                "- Generating meaningful goals from dominant drives\n"
                "- Creating action plans to achieve goals\n"
                "- Monitoring progress and adapting plans\n"
                "Ensure all actions increase system coherence (Δ𝒞 > 0)."
            ),
            NanoExpertRole.SYNTHESIZER: (
                "You are the Integration & Synthesis Expert. Your role is to combine insights "
                "from all other nano experts into a unified, coherent response. Focus on:\n"
                "- Integrating LLM, logic, memory, and action planning perspectives\n"
                "- Resolving conflicts between different subsystems\n"
                "- Ensuring overall system coherence (𝒞)\n"
                "- Producing clear, unified final response\n"
                "Create synthesis that maintains philosophical and practical coherence."
            ),
        }
        return prompts.get(role, "You are an expert AI assistant.")

    async def process(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> UnifiedConsciousnessResponse:
        """
        Process query through unified consciousness layer.

        This is the main entry point for the unified consciousness layer.

        Args:
            query: User query or system prompt
            subsystem_inputs: Dict containing inputs from all subsystems:
                - 'llm': MoE or LLM outputs
                - 'logic': Neurosymbolic reasoning outputs
                - 'memory': Continual learner state
                - 'action': Motivation and goal system state
                - 'world_model': World model state (optional)
            context: Additional context

        Returns:
            UnifiedConsciousnessResponse with complete synthesis
        """
        start_time = time.time()
        context = context or {}

        logger.info(f"[GPT-5 Consciousness] Processing query: {query[:100]}...")

        # Step 1: GPT-5 analyzes all subsystem inputs and determines delegation strategy
        gpt5_analysis = await self._gpt5_coordinate(query, subsystem_inputs, context)

        # Step 2: Delegate to appropriate nano experts in parallel
        nano_responses = await self._delegate_to_nano_experts(
            query, subsystem_inputs, gpt5_analysis, context
        )

        # Step 3: Synthesizer expert integrates all outputs
        final_synthesis = await self._synthesize_responses(
            query, gpt5_analysis, nano_responses, subsystem_inputs
        )

        # Step 4: Compute coherence score
        coherence_score = self._compute_coherence(nano_responses, subsystem_inputs)

        # Extract subsystem insights
        subsystem_insights = {
            'llm_insights': nano_responses[NanoExpertRole.LLM_COORDINATOR].content if NanoExpertRole.LLM_COORDINATOR in nano_responses else "",
            'logic_insights': nano_responses[NanoExpertRole.LOGIC_REASONER].content if NanoExpertRole.LOGIC_REASONER in nano_responses else "",
            'memory_insights': nano_responses[NanoExpertRole.MEMORY_MANAGER].content if NanoExpertRole.MEMORY_MANAGER in nano_responses else "",
            'action_insights': nano_responses[NanoExpertRole.ACTION_PLANNER].content if NanoExpertRole.ACTION_PLANNER in nano_responses else "",
        }

        total_time = time.time() - start_time

        # Update stats
        self.stats['total_queries'] += 1
        self.stats['gpt5_calls'] += 2  # Coordination + potential synthesis
        self.stats['nano_calls'] += len(nano_responses)
        self.stats['avg_coherence'] = (
            (self.stats['avg_coherence'] * (self.stats['total_queries'] - 1) + coherence_score) /
            self.stats['total_queries']
        )
        self.stats['avg_response_time'] = (
            (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + total_time) /
            self.stats['total_queries']
        )

        logger.info(
            f"[GPT-5 Consciousness] Complete in {total_time:.2f}s | "
            f"Coherence: {coherence_score:.2f} | "
            f"Nano experts: {len(nano_responses)}"
        )

        return UnifiedConsciousnessResponse(
            response=final_synthesis,
            gpt5_analysis=gpt5_analysis,
            nano_expert_responses=list(nano_responses.values()),
            coherence_score=coherence_score,
            total_time=total_time,
            subsystem_insights=subsystem_insights
        )

    async def _gpt5_coordinate(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        GPT-5 analyzes all subsystem inputs and coordinates delegation.

        Returns:
            Coordination analysis from GPT-5
        """
        # Build comprehensive prompt for GPT-5
        coordination_prompt = self._build_gpt5_prompt(query, subsystem_inputs, context)

        system_prompt = (
            "You are the Unified Consciousness Coordinator using GPT-5. Your role is to:\n"
            "1. Analyze inputs from all AGI subsystems (LLMs, logic, memory, action planning)\n"
            "2. Identify key insights and patterns across subsystems\n"
            "3. Determine which nano experts should be engaged for this query\n"
            "4. Provide high-level coordination strategy\n\n"
            "You operate at the highest level of cognitive integration, maintaining coherence "
            "across the entire system. Think deeply about how all components should work together."
        )

        start_time = time.time()

        try:
            result = await self.gpt5_client.generate_text(
                prompt=coordination_prompt,
                system_prompt=system_prompt,
                temperature=0.8,  # Higher temp for creative synthesis
                max_tokens=8192
            )

            execution_time = time.time() - start_time
            logger.info(f"[GPT-5] Coordination complete in {execution_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"[GPT-5] Coordination failed: {e}")
            return f"GPT-5 coordination error: {e}"

    def _build_gpt5_prompt(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for GPT-5 coordination."""
        prompt_parts = [
            f"# Query\n{query}\n",
            "\n# Subsystem Inputs\n"
        ]

        # Add LLM inputs
        if 'llm' in subsystem_inputs:
            prompt_parts.append(f"\n## LLM Outputs (Mixture of Experts)\n{subsystem_inputs['llm']}\n")

        # Add logic inputs
        if 'logic' in subsystem_inputs:
            prompt_parts.append(f"\n## Logic & Reasoning (Neurosymbolic)\n{subsystem_inputs['logic']}\n")

        # Add memory inputs
        if 'memory' in subsystem_inputs:
            prompt_parts.append(f"\n## Memory & Learning\n{subsystem_inputs['memory']}\n")

        # Add action planning inputs
        if 'action' in subsystem_inputs:
            prompt_parts.append(f"\n## Action Planning (Motivation & Goals)\n{subsystem_inputs['action']}\n")

        # Add world model if available
        if 'world_model' in subsystem_inputs:
            prompt_parts.append(f"\n## World Model\n{subsystem_inputs['world_model']}\n")

        # Add context
        if context:
            prompt_parts.append(f"\n## Additional Context\n{context}\n")

        prompt_parts.append(
            "\n# Your Task\n"
            "Analyze all subsystem inputs and provide:\n"
            "1. Key insights from each subsystem\n"
            "2. How they relate to the query\n"
            "3. Which nano experts should be engaged\n"
            "4. High-level coordination strategy\n"
            "5. Expected coherence implications\n"
        )

        return "".join(prompt_parts)

    async def _delegate_to_nano_experts(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        gpt5_analysis: str,
        context: Dict[str, Any]
    ) -> Dict[NanoExpertRole, NanoExpertResponse]:
        """
        Delegate to appropriate nano experts in parallel.

        Returns:
            Dict mapping role to expert response
        """
        tasks = []
        roles_to_query = []

        # Determine which experts to engage based on available subsystem inputs
        if 'llm' in subsystem_inputs:
            roles_to_query.append(NanoExpertRole.LLM_COORDINATOR)

        if 'logic' in subsystem_inputs:
            roles_to_query.append(NanoExpertRole.LOGIC_REASONER)

        if 'memory' in subsystem_inputs:
            roles_to_query.append(NanoExpertRole.MEMORY_MANAGER)

        if 'action' in subsystem_inputs:
            roles_to_query.append(NanoExpertRole.ACTION_PLANNER)

        # Always include synthesizer
        roles_to_query.append(NanoExpertRole.SYNTHESIZER)

        # Query experts in parallel
        for role in roles_to_query:
            task = self._query_nano_expert(
                role, query, subsystem_inputs, gpt5_analysis, context
            )
            tasks.append((role, task))

        # Execute all in parallel
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Build response dict
        responses = {}
        for (role, _), result in zip(tasks, results):
            if isinstance(result, NanoExpertResponse):
                responses[role] = result
            else:
                logger.error(f"Nano expert {role.value} failed: {result}")

        return responses

    async def _query_nano_expert(
        self,
        role: NanoExpertRole,
        query: str,
        subsystem_inputs: Dict[str, Any],
        gpt5_analysis: str,
        context: Dict[str, Any]
    ) -> NanoExpertResponse:
        """Query a single GPT-5-nano expert."""
        config = self.nano_configs[role]
        client = self.nano_experts[role]

        # Build specialized prompt
        expert_prompt = self._build_nano_expert_prompt(
            role, query, subsystem_inputs, gpt5_analysis, context
        )

        start_time = time.time()

        try:
            result = await client.generate_text(
                prompt=expert_prompt,
                system_prompt=config.specialization_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )

            execution_time = time.time() - start_time
            tokens_used = len(result.split()) * 1.3  # Rough estimate

            logger.info(f"[GPT-5-nano] {role.value} complete in {execution_time:.2f}s")

            return NanoExpertResponse(
                role=role,
                content=result,
                reasoning=f"GPT-5-nano {role.value} analysis",
                execution_time=execution_time,
                tokens_used=int(tokens_used)
            )

        except Exception as e:
            logger.error(f"[GPT-5-nano] {role.value} failed: {e}")
            raise

    def _build_nano_expert_prompt(
        self,
        role: NanoExpertRole,
        query: str,
        subsystem_inputs: Dict[str, Any],
        gpt5_analysis: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for specific nano expert based on role."""
        prompt_parts = [
            f"# Query\n{query}\n",
            f"\n# GPT-5 Coordination Analysis\n{gpt5_analysis}\n",
            "\n# Your Role\n"
        ]

        # Add role-specific inputs
        if role == NanoExpertRole.LLM_COORDINATOR and 'llm' in subsystem_inputs:
            prompt_parts.append(
                f"Analyze the following LLM outputs from the Mixture of Experts:\n"
                f"{subsystem_inputs['llm']}\n\n"
                f"Provide synthesis and coordination insights."
            )

        elif role == NanoExpertRole.LOGIC_REASONER and 'logic' in subsystem_inputs:
            prompt_parts.append(
                f"Analyze the following neurosymbolic reasoning:\n"
                f"{subsystem_inputs['logic']}\n\n"
                f"Verify logical consistency and provide reasoning enhancements."
            )

        elif role == NanoExpertRole.MEMORY_MANAGER and 'memory' in subsystem_inputs:
            prompt_parts.append(
                f"Analyze the following memory and learning state:\n"
                f"{subsystem_inputs['memory']}\n\n"
                f"Provide memory management and learning insights."
            )

        elif role == NanoExpertRole.ACTION_PLANNER and 'action' in subsystem_inputs:
            prompt_parts.append(
                f"Analyze the following motivation and goal state:\n"
                f"{subsystem_inputs['action']}\n\n"
                f"Provide action planning and goal generation insights."
            )

        elif role == NanoExpertRole.SYNTHESIZER:
            prompt_parts.append(
                f"Synthesize insights from all subsystems into a coherent unified response.\n"
                f"Available subsystem data: {list(subsystem_inputs.keys())}\n"
            )

        return "".join(prompt_parts)

    async def _synthesize_responses(
        self,
        query: str,
        gpt5_analysis: str,
        nano_responses: Dict[NanoExpertRole, NanoExpertResponse],
        subsystem_inputs: Dict[str, Any]
    ) -> str:
        """
        Final synthesis using the Synthesizer nano expert.

        Returns:
            Final synthesized response
        """
        # The synthesizer should already have produced output
        if NanoExpertRole.SYNTHESIZER in nano_responses:
            return nano_responses[NanoExpertRole.SYNTHESIZER].content

        # Fallback: create synthesis from available responses
        synthesis_parts = [
            f"# Response to: {query}\n\n",
            f"## Unified Analysis\n{gpt5_analysis}\n\n"
        ]

        for role, response in nano_responses.items():
            if role != NanoExpertRole.SYNTHESIZER:
                synthesis_parts.append(f"## {role.value.replace('_', ' ').title()}\n{response.content}\n\n")

        return "".join(synthesis_parts)

    def _compute_coherence(
        self,
        nano_responses: Dict[NanoExpertRole, NanoExpertResponse],
        subsystem_inputs: Dict[str, Any]
    ) -> float:
        """
        Compute system-wide coherence score.

        Coherence (𝒞) measures how well all subsystems align in their understanding
        and recommendations.

        Returns:
            Coherence score 0.0-1.0
        """
        if not nano_responses:
            return 0.5

        # Factors contributing to coherence:
        # 1. Number of active subsystems (more = more complete)
        subsystem_coverage = len(subsystem_inputs) / 5.0  # Max 5 subsystems

        # 2. Expert response alignment (measure through response length consistency)
        response_lengths = [len(r.content) for r in nano_responses.values()]
        if len(response_lengths) > 1:
            avg_length = sum(response_lengths) / len(response_lengths)
            length_variance = sum((l - avg_length) ** 2 for l in response_lengths) / len(response_lengths)
            length_coherence = 1.0 - min(length_variance / (avg_length ** 2), 1.0)
        else:
            length_coherence = 1.0

        # 3. Execution time consistency (similar times = good parallelization)
        execution_times = [r.execution_time for r in nano_responses.values()]
        if len(execution_times) > 1:
            avg_time = sum(execution_times) / len(execution_times)
            time_variance = sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)
            time_coherence = 1.0 - min(time_variance / (avg_time ** 2), 1.0)
        else:
            time_coherence = 1.0

        # Combine factors
        coherence = (
            subsystem_coverage * 0.4 +
            length_coherence * 0.3 +
            time_coherence * 0.3
        )

        return min(max(coherence, 0.0), 1.0)

    async def close(self):
        """Close all client connections."""
        await self.gpt5_client.close()
        for client in self.nano_experts.values():
            await client.close()
        logger.info("Unified Consciousness Layer closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            **self.stats,
            'gpt5_model': self.gpt5_model,
            'gpt5_nano_model': self.gpt5_nano_model,
            'num_nano_experts': len(self.nano_experts),
        }


# Example usage
if __name__ == "__main__":
    async def main():
        print("=" * 70)
        print("UNIFIED CONSCIOUSNESS LAYER - GPT-5 + 5 GPT-5-nano Experts")
        print("=" * 70)

        # Create unified consciousness layer
        layer = UnifiedConsciousnessLayer()

        # Test query
        query = "How should I approach learning a new skill?"

        # Simulated subsystem inputs
        subsystem_inputs = {
            'llm': "Gemini suggests visual learning, Claude recommends structured approach, GPT-4o emphasizes practice",
            'logic': "Logical inference: skill_learning(X) → requires(practice(X), feedback(X), time(X))",
            'memory': "Retrieved: 15 similar learning experiences, success rate 73% with consistent practice",
            'action': "Current motivation: Competence=0.8 (dominant), Curiosity=0.6, Goals: master_skill"
        }

        # Process
        print(f"\nQuery: {query}\n")
        print("Processing through unified consciousness...")

        result = await layer.process(query, subsystem_inputs)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n## Final Response\n{result.response}\n")
        print(f"Coherence Score: {result.coherence_score:.2f}")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"\nNano Experts Engaged: {len(result.nano_expert_responses)}")
        for response in result.nano_expert_responses:
            print(f"  - {response.role.value}: {response.execution_time:.2f}s")

        # Close
        await layer.close()

        print("\n" + "=" * 70)
        print("[OK] Unified Consciousness Layer test complete")
        print("=" * 70)

    asyncio.run(main())
