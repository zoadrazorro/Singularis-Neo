"""
MAIN BRAIN - Meta-level synthesis system

Collects outputs from all AGI subsystems and synthesizes them into
coherent session reports using GPT-4o.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemOutput:
    """Output from a single AGI subsystem."""
    system_name: str
    timestamp: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class SessionReport:
    """Complete session report."""
    session_id: str
    start_time: float
    end_time: float
    total_cycles: int
    outputs: List[SystemOutput]
    synthesis: str
    statistics: Dict[str, Any]


class MainBrain:
    """
    MAIN BRAIN - Synthesizes all AGI subsystem outputs.
    
    Uses GPT-4o to create coherent reports from distributed intelligence.
    """
    
    def __init__(self, openai_client, session_id: Optional[str] = None):
        """
        Initialize Main Brain.

        Args:
            openai_client: OpenAI client for GPT-4o
            session_id: Optional session ID (generated if not provided)
        """
        self.openai_client = openai_client
        self.session_id = session_id or self._generate_session_id()
        self.session_start = time.time()

        # Collect outputs from all subsystems
        self.outputs: List[SystemOutput] = []

        # Statistics
        self.total_cycles = 0
        self.system_activations: Dict[str, int] = {}

        # Affective tracking (emotional valence over session)
        self.valence_history: List[float] = []
        self.affect_history: List[str] = []
        self.active_affect_count: int = 0
        self.passive_affect_count: int = 0

        print(f"[MAIN BRAIN] 🧠 Initialized - Session ID: {self.session_id}")
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"skyrim_agi_{timestamp}_{unique_id}"
        
    def record_output(
        self,
        system_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> None:
        """Record output from a subsystem."""
        output = SystemOutput(
            system_name=system_name,
            timestamp=time.time(),
            content=content,
            metadata=metadata or {},
            success=success
        )
        
        self.outputs.append(output)
        
        # Update statistics
        if system_name not in self.system_activations:
            self.system_activations[system_name] = 0
        self.system_activations[system_name] += 1
        
    def increment_cycle(self) -> None:
        """Increment cycle counter."""
        self.total_cycles += 1

    def record_affective_state(
        self,
        valence: float,
        affect_type: str,
        is_active: bool
    ) -> None:
        """
        Record affective state for this cycle.

        Args:
            valence: Emotional valence value
            affect_type: Type of affect (joy, fear, etc.)
            is_active: True if active affect, False if passive
        """
        self.valence_history.append(valence)
        self.affect_history.append(affect_type)

        if is_active:
            self.active_affect_count += 1
        else:
            self.passive_affect_count += 1

    def get_affective_statistics(self) -> Dict[str, Any]:
        """
        Get affective statistics for the session.

        Returns:
            Dictionary with affective metrics
        """
        if not self.valence_history:
            return {
                'avg_valence': 0.0,
                'valence_range': (0.0, 0.0),
                'dominant_affects': [],
                'active_ratio': 0.0,
                'affective_volatility': 0.0
            }

        import numpy as np

        # Compute statistics
        avg_valence = float(np.mean(self.valence_history))
        min_valence = float(np.min(self.valence_history))
        max_valence = float(np.max(self.valence_history))
        valence_std = float(np.std(self.valence_history))

        # Dominant affects (top 3)
        affect_counts = {}
        for affect in self.affect_history:
            affect_counts[affect] = affect_counts.get(affect, 0) + 1

        sorted_affects = sorted(affect_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_affects = [affect for affect, _ in sorted_affects[:3]]

        # Active affect ratio
        total_affects = self.active_affect_count + self.passive_affect_count
        active_ratio = self.active_affect_count / total_affects if total_affects > 0 else 0.0

        return {
            'avg_valence': avg_valence,
            'valence_range': (min_valence, max_valence),
            'valence_std': valence_std,
            'dominant_affects': dominant_affects,
            'active_ratio': active_ratio,
            'affective_volatility': valence_std,
            'total_measurements': len(self.valence_history)
        }
        
    async def synthesize_report(self) -> str:
        """
        Synthesize all outputs into a coherent report using GPT-4o.
        """
        if not self.openai_client or not self.openai_client.is_available():
            return self._generate_fallback_report()
        
        # Prepare synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt()
        
        print("[MAIN BRAIN] 🧠 Synthesizing session report with GPT-4o...")
        
        try:
            synthesis = await self.openai_client.generate_text(
                prompt=synthesis_prompt,
                system_prompt="""You are the MAIN BRAIN of a Skyrim AGI system. 

Your role is to synthesize outputs from multiple AI subsystems (vision, reasoning, planning, sensorimotor, etc.) into a coherent, insightful narrative.

Create a comprehensive report that:
1. Summarizes key discoveries and patterns
2. Identifies successful system integrations
3. Highlights interesting behaviors and adaptations
4. Notes any issues or areas for improvement
5. Provides strategic recommendations

Write in a clear, technical style suitable for AI researchers. Be specific and reference actual system outputs.""",
                temperature=0.5,
                max_tokens=4096
            )
            
            print(f"[MAIN BRAIN] ✓ Synthesis complete ({len(synthesis)} chars)")
            return synthesis
            
        except Exception as e:
            print(f"[MAIN BRAIN] ⚠️ Synthesis failed: {e}")
            return self._generate_fallback_report()
            
    def _build_synthesis_prompt(self) -> str:
        """Build the synthesis prompt from all collected outputs."""
        
        # Session overview
        duration = time.time() - self.session_start
        duration_str = f"{duration/60:.1f} minutes"
        
        # Get affective statistics
        affective_stats = self.get_affective_statistics()

        prompt = f"""# Skyrim AGI Session Synthesis

## Session Information
- Session ID: {self.session_id}
- Duration: {duration_str}
- Total Cycles: {self.total_cycles}
- Systems Active: {len(self.system_activations)}

## Affective State (Emotional Valence)
- Average Valence: {affective_stats['avg_valence']:.3f}
- Valence Range: [{affective_stats['valence_range'][0]:.2f}, {affective_stats['valence_range'][1]:.2f}]
- Affective Volatility (σ): {affective_stats['affective_volatility']:.3f}
- Dominant Affects: {', '.join(affective_stats['dominant_affects']) if affective_stats['dominant_affects'] else 'None'}
- Active Affect Ratio: {affective_stats['active_ratio']:.1%}
- Total Affective Measurements: {affective_stats['total_measurements']}

## System Activations
"""
        
        # Add system activation counts
        sorted_systems = sorted(self.system_activations.items(), key=lambda x: x[1], reverse=True)
        for system, count in sorted_systems:
            prompt += f"- {system}: {count} activations\n"
        
        prompt += "\n## System Outputs\n\n"
        
        # Add recent outputs from each system (last 3 per system)
        outputs_by_system: Dict[str, List[SystemOutput]] = {}
        for output in self.outputs:
            if output.system_name not in outputs_by_system:
                outputs_by_system[output.system_name] = []
            outputs_by_system[output.system_name].append(output)
        
        for system, outputs in outputs_by_system.items():
            prompt += f"### {system}\n\n"
            
            # Take last 3 outputs
            recent = outputs[-3:]
            for output in recent:
                time_str = datetime.fromtimestamp(output.timestamp).strftime("%H:%M:%S")
                status = "✓" if output.success else "✗"
                
                # Truncate content if too long
                content = output.content[:500] + "..." if len(output.content) > 500 else output.content
                
                prompt += f"**[{time_str}] {status}**\n{content}\n\n"
        
        prompt += """
---

Please synthesize the above outputs into a coherent report. Focus on:

1. **Key Patterns:** What patterns emerged across systems?
2. **System Integration:** How well did different systems work together?
3. **Notable Behaviors:** Any interesting adaptations or discoveries?
4. **Performance:** Which systems performed well? Any issues?
5. **Recommendations:** What should be optimized or improved?

Provide a comprehensive but concise synthesis (aim for 1000-2000 words).
"""
        
        return prompt
        
    def _generate_fallback_report(self) -> str:
        """Generate simple text report if GPT-4o unavailable."""
        duration = time.time() - self.session_start
        
        report = f"""# Skyrim AGI Session Report (Fallback)

Session ID: {self.session_id}
Duration: {duration/60:.1f} minutes
Total Cycles: {self.total_cycles}

## System Activations
"""
        
        sorted_systems = sorted(self.system_activations.items(), key=lambda x: x[1], reverse=True)
        for system, count in sorted_systems:
            report += f"- {system}: {count} activations\n"
        
        report += f"\n## Total Outputs Collected: {len(self.outputs)}\n"
        report += "\n(GPT-4o synthesis unavailable - showing raw statistics)\n"
        
        return report
        
    async def generate_session_markdown(self, output_dir: str = "sessions") -> str:
        """
        Generate complete session report as markdown file.
        
        Returns:
            Path to generated markdown file
        """
        # Create sessions directory
        sessions_path = Path(output_dir)
        sessions_path.mkdir(exist_ok=True)
        
        # Generate synthesis
        synthesis = await self.synthesize_report()
        
        # Build complete markdown
        markdown = self._build_session_markdown(synthesis)
        
        # Save to file
        filename = f"{self.session_id}.md"
        filepath = sessions_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"[MAIN BRAIN] 📄 Session report saved: {filepath}")
        
        return str(filepath)
        
    def _build_session_markdown(self, synthesis: str) -> str:
        """Build complete markdown document."""
        
        start_dt = datetime.fromtimestamp(self.session_start)
        end_dt = datetime.now()
        duration = (end_dt - start_dt).total_seconds()
        
        markdown = f"""# Skyrim AGI Session Report

---

## Session Metadata

- **Session ID:** `{self.session_id}`
- **Start Time:** {start_dt.strftime("%Y-%m-%d %H:%M:%S")}
- **End Time:** {end_dt.strftime("%Y-%m-%d %H:%M:%S")}
- **Duration:** {duration/60:.1f} minutes ({duration:.0f} seconds)
- **Total Cycles:** {self.total_cycles}
- **Systems Active:** {len(self.system_activations)}
- **Outputs Collected:** {len(self.outputs)}

---

## GPT-4o Synthesis

{synthesis}

---

## System Activation Summary

| System | Activations | Success Rate |
|--------|-------------|--------------|
"""
        
        # Add system statistics
        for system, count in sorted(self.system_activations.items(), key=lambda x: x[1], reverse=True):
            # Calculate success rate
            system_outputs = [o for o in self.outputs if o.system_name == system]
            successes = sum(1 for o in system_outputs if o.success)
            success_rate = (successes / len(system_outputs) * 100) if system_outputs else 0
            
            markdown += f"| {system} | {count} | {success_rate:.1f}% |\n"
        
        markdown += "\n---\n\n## Detailed System Outputs\n\n"
        
        # Group outputs by system
        outputs_by_system: Dict[str, List[SystemOutput]] = {}
        for output in self.outputs:
            if output.system_name not in outputs_by_system:
                outputs_by_system[output.system_name] = []
            outputs_by_system[output.system_name].append(output)
        
        # Add each system's outputs
        for system in sorted(outputs_by_system.keys()):
            outputs = outputs_by_system[system]
            markdown += f"### {system} ({len(outputs)} outputs)\n\n"
            
            # Show last 5 outputs
            for output in outputs[-5:]:
                time_str = datetime.fromtimestamp(output.timestamp).strftime("%H:%M:%S")
                status = "✅ Success" if output.success else "❌ Failed"
                
                markdown += f"#### [{time_str}] {status}\n\n"
                markdown += f"```\n{output.content[:1000]}\n```\n\n"
                
                if output.metadata:
                    markdown += f"**Metadata:** {output.metadata}\n\n"
        
        markdown += "\n---\n\n"
        markdown += f"*Report generated by MAIN BRAIN at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return markdown
