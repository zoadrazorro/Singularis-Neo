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
    """A user instruction given to the AGI."""
    text: str
    timestamp: datetime
    priority: str  # 'high', 'medium', 'low'
    context: Dict[str, Any]  # Game state when instruction was given
    active: bool = True
    fulfilled: bool = False


class InstructionSystem:
    """
    Manages user instructions and integrates them into AGI decision-making.
    
    Features:
    - Real-time instruction queue
    - Priority-based instruction handling
    - Context-aware instruction interpretation
    - Instruction memory and learning
    - Feedback loop for instruction effectiveness
    """
    
    def __init__(self, max_active_instructions: int = 5):
        """
        Initialize instruction system.
        
        Args:
            max_active_instructions: Max number of active instructions to consider
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
        """Start background task for console input."""
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
        """Stop console input task."""
        self.running = False
        if self.input_task:
            self.input_task.cancel()
    
    async def _console_input_loop(self):
        """Background loop for reading console input."""
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
        """
        Add a new instruction from the user.
        
        Args:
            text: Instruction text
            priority: 'high', 'medium', or 'low'
            context: Current game state context
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
        """Update list of active instructions based on priority and relevance."""
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
        """
        Get formatted instruction context for LLM reasoning.
        
        Args:
            current_state: Current game state
            
        Returns:
            Formatted instruction context string
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
        """
        Mark an instruction as fulfilled.
        
        Args:
            instruction_text: Text of the instruction
            success: Whether it was successfully fulfilled
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
        """Clear all active instructions."""
        self.active_instructions.clear()
        self.instruction_queue.clear()
        print("[INSTRUCTION] 🗑️ All instructions cleared")
    
    def list_instructions(self):
        """Print all active instructions."""
        if not self.active_instructions:
            print("[INSTRUCTION] No active instructions")
            return
        
        print(f"\n[INSTRUCTION] Active instructions ({len(self.active_instructions)}):")
        for i, instr in enumerate(self.active_instructions, 1):
            age = (datetime.now() - instr.timestamp).total_seconds()
            print(f"  {i}. [{instr.priority}] {instr.text} ({age:.0f}s ago)")
        print()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get instruction system statistics."""
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
