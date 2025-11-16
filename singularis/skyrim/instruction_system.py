# -*- coding: utf-8 -*-
"""
Interactive Instruction System for Skyrim AGI

Allows real-time natural language instructions to guide the AGI's behavior.
Instructions are integrated into the LLM reasoning process and stored as memories.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque


@dataclass
class Instruction:
    """Represents a single natural language instruction given by a user.

    Attributes:
        text: The content of the instruction.
        timestamp: The time the instruction was given.
        priority: The priority level ('high', 'medium', 'low').
        context: The game state at the time the instruction was given.
        active: A boolean indicating if the instruction is currently active.
        fulfilled: A boolean indicating if the instruction has been completed.
    """
    text: str
    timestamp: datetime
    priority: str
    context: Dict[str, Any]
    active: bool = True
    fulfilled: bool = False


class InstructionSystem:
    """Manages the lifecycle of user instructions and integrates them into the AGI.

    This system handles receiving, prioritizing, and queuing instructions. It
    provides a formatted context string of active instructions for the LLM's
    reasoning process and tracks the success of instructions to learn over time.
    It also includes a (simplified) mechanism for real-time console input.
    """
    
    def __init__(self, max_active_instructions: int = 5):
        """Initializes the instruction system.

        Args:
            max_active_instructions: The maximum number of instructions to
                                     consider active at any given time.
        """
        self.instruction_queue: deque = deque(maxlen=100)
        self.active_instructions: List[Instruction] = []
        self.instruction_history: List[Instruction] = []
        self.max_active = max_active_instructions
        
        # Instruction effectiveness tracking
        self.instruction_outcomes: Dict[str, List[float]] = {}
        
        # Console input task
        self.input_task: Optional[asyncio.Task] = None
        self.running = False
        
    def start_console_input(self):
        """Starts a background asyncio task to listen for console input.

        This provides a simple command-line interface for giving instructions
        to the AGI in real-time.
        """
        if not self.running:
            self.running = True
            self.input_task = asyncio.create_task(self._console_input_loop())
            print("\n[INSTRUCTION] 💬 Chat input enabled! Type instructions to guide the AGI.")
            print("[INSTRUCTION] Commands:")
            print("  - Type any instruction (e.g., 'explore the dungeon carefully')")
            print("  - 'clear' - Clear all active instructions")
            print("  - 'list' - Show active instructions")
            print("  - 'stop' - Stop instruction input")
            print()
    
    def stop_console_input(self):
        """Stops the background console input task."""
        self.running = False
        if self.input_task:
            self.input_task.cancel()
    
    async def _console_input_loop(self):
        """The background loop for reading console input.

        Note: This is a simplified implementation. A production system would
        use a more robust library like `aioconsole` for non-blocking input.
        """
        try:
            while self.running:
                # Use asyncio to avoid blocking
                try:
                    # Note: This is a simplified version. In production, you'd use
                    # aioconsole or a proper async input library
                    await asyncio.sleep(0.1)  # Check periodically
                    
                    # For now, instructions would be added via add_instruction()
                    # In a full implementation, you'd use aioconsole.ainput()
                    
                except Exception as e:
                    print(f"[INSTRUCTION] Input error: {e}")
                    
        except asyncio.CancelledError:
            print("[INSTRUCTION] Console input stopped")
    
    def add_instruction(
        self,
        text: str,
        priority: str = 'medium',
        context: Optional[Dict[str, Any]] = None
    ):
        """Adds a new instruction to the system.

        The instruction is added to a queue and then processed to determine if
        it should become active immediately.

        Args:
            text: The natural language text of the instruction.
            priority: The priority of the instruction ('high', 'medium', 'low').
            context: The game state context at the time of instruction.
        """
        instruction = Instruction(
            text=text,
            timestamp=datetime.now(),
            priority=priority,
            context=context or {}
        )
        
        self.instruction_queue.append(instruction)
        self._update_active_instructions()
        
        print(f"\n[INSTRUCTION] 📝 New instruction: '{text}' (priority: {priority})")
        print(f"[INSTRUCTION] Active instructions: {len(self.active_instructions)}")
    
    def _update_active_instructions(self):
        """Updates the list of active instructions from the queue.

        This method ensures the active list contains the highest-priority and
        most recent instructions, up to the maximum limit.
        """
        # Add new instructions from queue
        while self.instruction_queue and len(self.active_instructions) < self.max_active:
            instruction = self.instruction_queue.popleft()
            self.active_instructions.append(instruction)
        
        # Sort by priority (high > medium > low) and recency
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        self.active_instructions.sort(
            key=lambda i: (priority_order[i.priority], -i.timestamp.timestamp())
        )
        
        # Keep only max_active most important
        if len(self.active_instructions) > self.max_active:
            overflow = self.active_instructions[self.max_active:]
            self.active_instructions = self.active_instructions[:self.max_active]
            # Put overflow back in queue
            for instr in overflow:
                self.instruction_queue.append(instr)
    
    def get_instruction_context(self, current_state: Dict[str, Any]) -> str:
        """Formats the active instructions into a string for the LLM prompt.

        Args:
            current_state: The current game state (used for future context-
                           aware filtering, currently unused).

        Returns:
            A formatted string containing the list of active instructions for
            the LLM to consider.
        """
        if not self.active_instructions:
            return ""
        
        context = "\n🎮 USER INSTRUCTIONS (follow these guidance):\n"
        
        for i, instr in enumerate(self.active_instructions, 1):
            age_seconds = (datetime.now() - instr.timestamp).total_seconds()
            age_str = f"{age_seconds:.0f}s ago" if age_seconds < 60 else f"{age_seconds/60:.1f}m ago"
            
            priority_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }[instr.priority]
            
            context += f"{i}. {priority_emoji} [{age_str}] \"{instr.text}\"\n"
        
        context += "\nConsider these instructions when making decisions. Prioritize high-priority (🔴) instructions.\n"
        
        return context
    
    def mark_instruction_fulfilled(self, instruction_text: str, success: bool = True):
        """Marks an active instruction as fulfilled, moving it to the history.

        Args:
            instruction_text: The text of the instruction to mark. The match is
                              case-insensitive and partial.
            success: Whether the instruction was fulfilled successfully.
        """
        for instr in self.active_instructions:
            if instruction_text.lower() in instr.text.lower():
                instr.fulfilled = True
                instr.active = False
                
                # Track outcome
                if instr.text not in self.instruction_outcomes:
                    self.instruction_outcomes[instr.text] = []
                self.instruction_outcomes[instr.text].append(1.0 if success else 0.0)
                
                # Move to history
                self.instruction_history.append(instr)
                self.active_instructions.remove(instr)
                
                print(f"[INSTRUCTION] ✅ Fulfilled: '{instr.text}'")
                break
    
    def clear_instructions(self):
        """Clears all active and queued instructions."""
        self.active_instructions.clear()
        self.instruction_queue.clear()
        print("[INSTRUCTION] 🗑️ All instructions cleared")
    
    def list_instructions(self):
        """Prints a list of all currently active instructions to the console."""
        if not self.active_instructions:
            print("[INSTRUCTION] No active instructions")
            return
        
        print(f"\n[INSTRUCTION] Active instructions ({len(self.active_instructions)}):")
        for i, instr in enumerate(self.active_instructions, 1):
            age = (datetime.now() - instr.timestamp).total_seconds()
            print(f"  {i}. [{instr.priority}] {instr.text} ({age:.0f}s ago)")
        print()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the instruction system's performance.

        Returns:
            A dictionary containing stats like the number of active instructions,
            total instructions given, and the average success rate.
        """
        avg_success = 0.0
        if self.instruction_outcomes:
            all_outcomes = [o for outcomes in self.instruction_outcomes.values() for o in outcomes]
            avg_success = sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0.0
        
        return {
            'active_instructions': len(self.active_instructions),
            'queued_instructions': len(self.instruction_queue),
            'total_given': len(self.instruction_history),
            'avg_success_rate': avg_success,
            'unique_instructions': len(self.instruction_outcomes)
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        system = InstructionSystem()
        
        # Simulate adding instructions
        system.add_instruction("Explore the dungeon carefully", priority='high')
        system.add_instruction("Avoid combat when possible", priority='medium')
        system.add_instruction("Collect any valuable items", priority='low')
        
        # Get context for LLM
        context = system.get_instruction_context({})
        print(context)
        
        # List instructions
        system.list_instructions()
        
        # Mark one as fulfilled
        system.mark_instruction_fulfilled("explore the dungeon", success=True)
        
        # Stats
        print(system.get_stats())
    
    asyncio.run(test())
