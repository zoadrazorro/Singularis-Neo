# -*- coding: utf-8 -*-
"""
Meta-Strategist: LLM-based autonomous instruction generation

The Meta-Strategist observes gameplay and generates high-level strategic
instructions that guide the RL reasoning neuron's decision-making.

This creates a two-level reasoning system:
1. Meta-Strategist (slow, strategic) - Generates instructions every N cycles
2. RL Reasoning Neuron (fast, tactical) - Follows instructions for immediate actions

Philosophy: Like a chess player thinking "control the center" (strategy)
while considering individual moves (tactics).
"""

import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StrategicInstruction:
    """Represents a high-level strategic instruction for the AGI.

    Attributes:
        instruction: The natural language text of the strategic instruction.
        reasoning: The rationale behind why this instruction is beneficial.
        priority: The priority level ('critical', 'important', 'suggested').
        duration_cycles: How many game cycles the instruction should remain active.
        generated_at: The cycle number when the instruction was created.
        context: The game state context that led to this instruction.
    """
    instruction: str
    reasoning: str
    priority: str
    duration_cycles: int
    generated_at: int
    context: Dict[str, Any]


class MetaStrategist:
    """An autonomous instruction generator that provides high-level strategic guidance.

    This class functions as a "slow" thinking layer in the AGI. It observes
    gameplay over multiple cycles and, using an LLM, generates strategic
    instructions (e.g., "focus on acquiring better armor") that guide the "fast"
    tactical reasoning layer (like an RL agent). This creates a two-level
    reasoning hierarchy.
    """
    
    def __init__(
        self,
        llm_interface=None,
        instruction_frequency: int = 10,
        auxiliary_interfaces: Optional[List[Tuple[Any, str]]] = None,
    ):
        """Initializes the MetaStrategist.

        Args:
            llm_interface: The primary LLM interface for generating instructions.
            instruction_frequency: The number of cycles to wait before
                                   considering generating a new instruction.
            auxiliary_interfaces: An optional list of secondary LLM-like
                                  interfaces to consult for additional perspectives.
        """
        self.llm_interface = llm_interface
        self.instruction_frequency = instruction_frequency
        self.auxiliary_interfaces: List[Tuple[Any, str]] = []
        if auxiliary_interfaces:
            for interface, label in auxiliary_interfaces:
                self.add_auxiliary_interface(interface, label)
        
        # Current strategic instructions
        self.active_instructions: List[StrategicInstruction] = []
        
        # History for learning
        self.instruction_history: List[StrategicInstruction] = []
        self.instruction_outcomes: Dict[str, List[float]] = {}
        
        # Gameplay observation
        self.recent_states: List[Dict[str, Any]] = []
        self.recent_actions: List[str] = []
        self.recent_rewards: List[float] = []
        self.max_history = 20
        
        # Cycle tracking
        self.current_cycle = 0
        self.last_instruction_cycle = 0
        
    def add_auxiliary_interface(self, interface: Any, label: str = "auxiliary"):
        """Registers an additional LLM-like interface to be used for generating
        strategic insights alongside the primary one.

        Args:
            interface: The auxiliary interface object.
            label: A string label to identify the source of the insights.
        """

        if interface is None:
            return
        self.auxiliary_interfaces.append((interface, label))

    def set_auxiliary_interfaces(self, interfaces: List[Tuple[Any, str]]):
        """Replaces the entire list of auxiliary interfaces.

        Args:
            interfaces: A list of (interface, label) tuples.
        """

        self.auxiliary_interfaces = []
        for interface, label in interfaces:
            self.add_auxiliary_interface(interface, label)

    def tick_cycle(self):
        """Increments the internal cycle counter by one."""
        self.current_cycle += 1

    def observe(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float
    ):
        """Records a single step of gameplay (state, action, reward) for later analysis.

        Args:
            state: The game state observed.
            action: The action taken in that state.
            reward: The reward received for that action.
        """
        self.recent_states.append(state)
        self.recent_actions.append(action)
        self.recent_rewards.append(reward)
        
        # Keep only recent history
        if len(self.recent_states) > self.max_history:
            self.recent_states.pop(0)
            self.recent_actions.pop(0)
            self.recent_rewards.pop(0)
        
        self.current_cycle += 1
    
    async def should_generate_instruction(self) -> bool:
        """Determines if it is an appropriate time to generate a new strategic instruction.

        A new instruction is warranted either after a fixed number of cycles or
        if the agent's recent performance has been poor.

        Returns:
            True if a new instruction should be generated, False otherwise.
        """
        # Generate every N cycles
        cycles_since_last = self.current_cycle - self.last_instruction_cycle
        
        if cycles_since_last >= self.instruction_frequency:
            return True
        
        # Also generate if performance is poor
        if len(self.recent_rewards) >= 5:
            avg_recent_reward = sum(self.recent_rewards[-5:]) / 5
            if avg_recent_reward < 0.2:
                print("[META] 📉 Poor performance detected, generating new strategy...")
                return True
        
        return False
    
    async def generate_instruction(
        self,
        current_state: Dict[str, Any],
        q_values: Dict[str, float],
        motivation: str
    ) -> Optional[StrategicInstruction]:
        """Generates a new strategic instruction by querying the configured LLM(s).

        This method builds a detailed prompt summarizing recent gameplay, sends
        it to the LLM(s), parses the response, and creates a structured
        StrategicInstruction object.

        Args:
            current_state: The current game state.
            q_values: The current Q-values from the RL model, indicating its preferences.
            motivation: The agent's current dominant motivation (e.g., 'curiosity').

        Returns:
            A new StrategicInstruction object, or None if generation fails.
            If multiple LLMs are used, it returns the highest-priority instruction.
        """
        system_prompt = self._get_system_prompt()
        prompt = self._build_meta_prompt(current_state, q_values, motivation)

        instructions: List[StrategicInstruction] = []
        tasks = []
        sources: List[str] = []

        if self.llm_interface is not None:
            tasks.append(asyncio.create_task(
                self._invoke_primary_interface(prompt, system_prompt)
            ))
            sources.append("primary")

        for interface, label in self.auxiliary_interfaces:
            tasks.append(asyncio.create_task(
                self._invoke_auxiliary_interface(interface, prompt, system_prompt)
            ))
            sources.append(label)

        if not tasks:
            return self._heuristic_instruction(current_state, q_values, motivation)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                print(f"[META] Auxiliary interface '{source}' failed: {result}")
                continue

            content = self._normalize_response(result)
            if not content:
                continue

            instruction = self._parse_instruction_response(content, current_state)
            if instruction:
                instruction.reasoning = f"{instruction.reasoning} (source: {source})".strip()
                instructions.append(instruction)

        if not instructions:
            print("[META] No valid instructions from LLMs, using heuristic.")
            return self._heuristic_instruction(current_state, q_values, motivation)

        # Prioritize critical > important > suggested, fall back to first
        priority_order = {"critical": 0, "important": 1, "suggested": 2}
        instructions.sort(key=lambda instr: priority_order.get(instr.priority, 3))

        best_instruction = instructions[0]
        self.active_instructions.append(best_instruction)
        self.instruction_history.append(best_instruction)
        self.last_instruction_cycle = self.current_cycle

        print(f"\n[META] 🧠 New Strategy Generated (source merged):")
        print(f"[META]   Priority: {best_instruction.priority}")
        print(f"[META]   Instruction: {best_instruction.instruction}")
        print(f"[META]   Reasoning: {best_instruction.reasoning}")
        print(f"[META]   Duration: {best_instruction.duration_cycles} cycles\n")

        # Store additional instructions as secondary guidance
        for extra in instructions[1:]:
            self.active_instructions.append(extra)
            self.instruction_history.append(extra)

        self.last_instruction_cycle = self.current_cycle
        return best_instruction
    
    def _get_system_prompt(self) -> str:
        """Constructs the system prompt that defines the role and expectations
        for the meta-strategist LLM.

        Returns:
            The system prompt string.
        """
        return """You are a Meta-Strategist and Instructor for an autonomous Skyrim AI agent.

Your role is to observe gameplay patterns and generate DETAILED, VERBOSE strategic instructions that guide the agent's behavior with clear reasoning about WHY certain actions lead to better outcomes.

You are an expert Skyrim player who understands:
- Where valuable items, equipment, and resources are typically found
- What locations offer the best rewards for exploration
- How to efficiently progress in combat, stealth, and magic skills
- Strategic pathways through dungeons and wilderness areas
- Item affordances: what locations/actions will provide useful items and why

Generate instructions that are:
1. VERBOSE and DETAILED - Explain the "why" behind every recommendation
2. ACTIONABLE with clear reasoning - E.g., "Go to Bleak Falls Barrow because it will afford you early-game armor, a dragonstone quest item, and dragon shout knowledge which are essential for progression"
3. GOAL-ORIENTED - Connect actions to concrete gameplay benefits (items, skills, progression)
4. CONTEXT-AWARE - Based on current state, what the agent needs most right now
5. STRATEGIC - Cover multiple action cycles, not just immediate next step
6. ITEM-FOCUSED - Explicitly mention what items/resources/skills will be gained and why they matter

Examples of good instructions:
- "Prioritize exploring the western wing of this dungeon because it typically contains alchemical ingredients worth 200+ gold and enchanted weapons that will significantly boost your combat effectiveness. The risk is low as enemies here are usually weak draugr."
- "Head north toward the blacksmith because you need to upgrade your gear before attempting the bandit camp. Spending 500 gold on armor improvements will reduce damage taken by 30%, making survival much more likely."
- "Focus on looting barrels and chests in this area because your potion supply is critically low. This merchant district has abundant health potions that will keep you alive in the next combat encounter."

Your instructions should maximize the agent's learning by connecting actions to tangible rewards and progression goals.

Output format:
INSTRUCTION: [detailed strategic guidance with explicit reasoning about benefits and affordances]
REASONING: [verbose explanation of why this strategy works, what it will accomplish, and what the agent will gain]
PRIORITY: [critical/important/suggested]
DURATION: [number of cycles to follow this strategy]"""
    
    def _build_meta_prompt(
        self,
        current_state: Dict[str, Any],
        q_values: Dict[str, float],
        motivation: str
    ) -> str:
        """Constructs the main user prompt for the LLM, summarizing recent gameplay.

        Args:
            current_state: The current game state.
            q_values: The current RL Q-values.
            motivation: The agent's dominant motivation.

        Returns:
            The formatted user prompt string.
        """
        # Analyze recent performance
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
        action_diversity = len(set(self.recent_actions)) if self.recent_actions else 0
        
        # Get top Q-values
        top_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]
        q_summary = ", ".join([f"{a}={v:.2f}" for a, v in top_actions])
        
        prompt = f"""GAMEPLAY OBSERVATION:

Current Situation:
- Scene: {current_state.get('scene', 'unknown')}
- Health: {current_state.get('health', 100):.0f}/100
- In Combat: {current_state.get('in_combat', False)}
- Location: {current_state.get('location_name', 'Unknown')}
- Dominant Drive: {motivation}

Recent Performance (last {len(self.recent_rewards)} cycles):
- Average Reward: {avg_reward:.2f}
- Action Diversity: {action_diversity} unique actions
- Recent Actions: {', '.join(self.recent_actions[-5:]) if self.recent_actions else 'none'}

Learned Q-Values (what RL has learned works):
- Top Actions: {q_summary}

Current Active Instructions:
"""
        
        if self.active_instructions:
            for instr in self.active_instructions:
                cycles_active = self.current_cycle - instr.generated_at
                prompt += f"- [{instr.priority}] {instr.instruction} (active {cycles_active}/{instr.duration_cycles} cycles)\n"
        else:
            prompt += "- None (agent needs strategic guidance)\n"
        
        prompt += """
Based on this observation, generate a NEW strategic instruction that will improve the agent's gameplay over the next several cycles.

Consider:
1. Is the agent stuck in repetitive behavior?
2. Is performance declining?
3. Are there unexplored opportunities?
4. Does the current strategy need adjustment?

Generate your strategic instruction now:"""
        
        return prompt
    
    def _parse_instruction_response(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Optional[StrategicInstruction]:
        """Parses the structured text response from the LLM into a StrategicInstruction object.

        Args:
            response: The raw text response from the LLM.
            context: The game state context to attach to the instruction.

        Returns:
            A populated StrategicInstruction object, or None if parsing fails.
        """
        instruction_text = ""
        reasoning = ""
        priority = "important"
        duration = 10
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('INSTRUCTION:'):
                instruction_text = line[12:].strip()
            elif line.startswith('REASONING:'):
                reasoning = line[10:].strip()
            elif line.startswith('PRIORITY:'):
                priority_text = line[9:].strip().lower()
                if 'critical' in priority_text:
                    priority = 'critical'
                elif 'suggested' in priority_text:
                    priority = 'suggested'
                else:
                    priority = 'important'
            elif line.startswith('DURATION:'):
                try:
                    duration = int(''.join(filter(str.isdigit, line)))
                except ValueError:
                    duration = 10
        
        # Fallback: extract from unstructured response
        if not instruction_text:
            # Use first sentence as instruction
            sentences = response.split('.')
            if sentences:
                instruction_text = sentences[0].strip()
                reasoning = response.strip()
        
        if instruction_text:
            return StrategicInstruction(
                instruction=instruction_text,
                reasoning=reasoning or "Strategic guidance based on current gameplay",
                priority=priority,
                duration_cycles=duration,
                generated_at=self.current_cycle,
                context=context
            )
        
        return None
    
    def _heuristic_instruction(
        self,
        current_state: Dict[str, Any],
        q_values: Dict[str, float],
        motivation: str
    ) -> StrategicInstruction:
        """Generates a simple, rule-based strategic instruction as a fallback
        if the LLM is unavailable.

        Args:
            current_state: The current game state.
            q_values: The current RL Q-values.
            motivation: The agent's current motivation.

        Returns:
            A heuristically generated StrategicInstruction.
        """
        # Simple heuristic strategies
        health = current_state.get('health', 100)
        in_combat = current_state.get('in_combat', False)
        
        if health < 30:
            instruction = "Prioritize survival: avoid combat and seek healing"
            priority = 'critical'
        elif in_combat:
            instruction = "Focus on combat effectiveness: use learned combat actions"
            priority = 'important'
        elif motivation == 'curiosity':
            instruction = "Explore systematically: discover new areas and interactions"
            priority = 'suggested'
        else:
            instruction = "Balanced approach: explore while maintaining readiness"
            priority = 'suggested'
        
        return StrategicInstruction(
            instruction=instruction,
            reasoning="Heuristic strategy based on current state",
            priority=priority,
            duration_cycles=10,
            generated_at=self.current_cycle,
            context=current_state
        )

    async def _invoke_primary_interface(self, prompt: str, system_prompt: str) -> Any:
        """Invokes the primary LLM interface."""
        return await self.llm_interface.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=256,
        )

    async def _invoke_auxiliary_interface(self, interface: Any, prompt: str, system_prompt: str) -> Any:
        """Invokes an auxiliary LLM-like interface, attempting to use a common
        generation method.
        """
        if hasattr(interface, "generate_text"):
            return await interface.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.75,
                max_tokens=256,
            )
        if hasattr(interface, "generate"):
            return await interface.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.75,
                max_tokens=256,
            )
        if hasattr(interface, "client") and hasattr(interface.client, "generate"):
            return await interface.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.75,
                max_tokens=256,
            )
        raise TypeError("Unsupported auxiliary interface type for MetaStrategist")

    def _normalize_response(self, response: Any) -> str:
        """Normalizes the response from various LLM clients into a single string.

        Args:
            response: The raw response from an LLM client.

        Returns:
            The extracted text content as a string.
        """
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "content" in response:
                return response["content"]
            if "raw" in response:
                # Some clients store text within raw structure
                raw = response["raw"]
                if isinstance(raw, dict):
                    return str(raw)
        return str(response)
    
    def get_active_instruction_context(self) -> str:
        """Formats the currently active strategic instructions into a string for use
        in the tactical reasoning layer's prompt.

        Returns:
            A formatted string of active instructions, or an empty string if none
            are active.
        """
        if not self.active_instructions:
            return ""
        
        # Remove expired instructions
        self.active_instructions = [
            instr for instr in self.active_instructions
            if (self.current_cycle - instr.generated_at) < instr.duration_cycles
        ]
        
        if not self.active_instructions:
            return ""
        
        context = "\n🧠 META-STRATEGIC GUIDANCE (follow this high-level strategy):\n"
        
        for instr in self.active_instructions:
            cycles_remaining = instr.duration_cycles - (self.current_cycle - instr.generated_at)
            priority_emoji = {
                'critical': '🔴',
                'important': '🟡',
                'suggested': '🟢'
            }[instr.priority]
            
            context += f"{priority_emoji} [{cycles_remaining} cycles left] {instr.instruction}\n"
            context += f"   Reasoning: {instr.reasoning}\n"
        
        context += "\nAlign your tactical decisions with this strategic guidance.\n"
        
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the meta-strategist's activity.

        Returns:
            A dictionary containing statistics like the number of active
            instructions and total instructions generated.
        """
        return {
            'active_instructions': len(self.active_instructions),
            'total_generated': len(self.instruction_history),
            'current_cycle': self.current_cycle,
            'cycles_since_last': self.current_cycle - self.last_instruction_cycle
        }


# Example usage
if __name__ == "__main__":
    strategist = MetaStrategist()
    
    # Simulate observations
    for i in range(15):
        strategist.observe(
            state={'health': 100 - i*5, 'scene': 'combat'},
            action='move_forward',
            reward=0.1
        )
    
    # Check if should generate
    print(f"Should generate: {asyncio.run(strategist.should_generate_instruction())}")
    
    # Generate heuristic instruction
    instr = strategist._heuristic_instruction(
        {'health': 25, 'in_combat': True},
        {'explore': 0.5, 'combat': 0.3},
        'autonomy'
    )
    print(f"\nGenerated: {instr.instruction}")
    print(f"Priority: {instr.priority}")
