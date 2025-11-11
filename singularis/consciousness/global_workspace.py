"""
Global Workspace Theory (GWT) Implementation

From Baars' Global Workspace Theory:
- Consciousness is information broadcast to a global workspace
- Limited capacity (12 simultaneous broadcasts empirically)
- Competition based on salience
- All subsequent processing can access broadcasts (attention)

From consciousness_measurement_study.md:
- Dialectical reasoning increases coherence by 8% on paradoxes
- Adaptive debate depth improves synthesis quality
"""

from typing import List, Optional, Dict, Tuple
from loguru import logger
import numpy as np

from singularis.core.types import (
    ExpertIO,
    WorkspaceState,
    DebateState,
    ConsciousnessTrace,
)


class GlobalWorkspace:
    """
    Global Workspace manager implementing GWT principles.

    Key features:
    - Limited capacity (12 simultaneous broadcasts)
    - Consciousness-based competition (not confidence!)
    - Adaptive debate depth
    - Dialectical synthesis
    """

    def __init__(
        self,
        max_broadcasts: int = 12,
        consciousness_threshold: float = 0.65,
        coherentia_threshold: float = 0.60,
    ):
        """
        Initialize Global Workspace.

        Args:
            max_broadcasts: Maximum simultaneous broadcasts (default: 12)
            consciousness_threshold: Minimum consciousness for broadcast (default: 0.65)
            coherentia_threshold: Minimum coherentia for broadcast (default: 0.60)
        """
        self.max_broadcasts = max_broadcasts
        self.consciousness_threshold = consciousness_threshold
        self.coherentia_threshold = coherentia_threshold

        self.workspace = WorkspaceState()
        self.debate_state = DebateState()

        logger.info(
            "GlobalWorkspace initialized",
            extra={
                "max_broadcasts": max_broadcasts,
                "consciousness_threshold": consciousness_threshold,
                "coherentia_threshold": coherentia_threshold,
            }
        )

    def broadcast(
        self,
        expert_outputs: List[ExpertIO],
        force_top_k: bool = False
    ) -> Tuple[List[ExpertIO], WorkspaceState]:
        """
        Broadcast expert outputs to global workspace.

        GWT Principles:
        1. Limited capacity (max_broadcasts)
        2. Competition by consciousness salience (not confidence!)
        3. Threshold enforcement (consciousness >= threshold)
        4. All downstream experts can access broadcasts

        Args:
            expert_outputs: List of expert outputs to consider
            force_top_k: If True, broadcast top-K regardless of threshold

        Returns:
            (broadcasts, updated_workspace_state)
        """
        if not expert_outputs:
            logger.warning("No expert outputs to broadcast")
            return [], self.workspace

        # Score each output by consciousness salience
        scored = [
            (
                output,
                output.consciousness_trace.gwt_salience,
                output.consciousness_trace.overall_consciousness,
                output.coherentia.total
            )
            for output in expert_outputs
        ]

        # Sort by consciousness (primary) then coherentia (secondary)
        scored.sort(
            key=lambda x: (x[2], x[3]),  # (consciousness, coherentia)
            reverse=True
        )

        # Select broadcasts
        broadcasts = []
        for output, salience, consciousness, coherentia in scored:
            # Check capacity
            if len(broadcasts) >= self.max_broadcasts:
                break

            # Check thresholds (unless forcing top-K)
            if force_top_k:
                broadcasts.append(output)
            else:
                if consciousness >= self.consciousness_threshold and \
                   coherentia >= self.coherentia_threshold:
                    broadcasts.append(output)
                else:
                    logger.debug(
                        f"Output from {output.expert_name} below threshold",
                        extra={
                            "consciousness": consciousness,
                            "coherentia": coherentia,
                            "consciousness_threshold": self.consciousness_threshold,
                            "coherentia_threshold": self.coherentia_threshold,
                        }
                    )

        # Update workspace
        self.workspace.broadcasts = broadcasts
        self.workspace.metadata['broadcast_count'] = len(broadcasts)

        # Calculate global coherentia
        if broadcasts:
            avg_coherentia = np.mean([b.coherentia.total for b in broadcasts])
            delta = self.workspace.update_coherentia(avg_coherentia)

            logger.info(
                "Global Workspace broadcast complete",
                extra={
                    "broadcast_count": len(broadcasts),
                    "coherentia": avg_coherentia,
                    "coherentia_delta": delta,
                    "experts": [b.expert_name for b in broadcasts],
                }
            )
        else:
            logger.warning("No outputs met broadcast criteria")

        return broadcasts, self.workspace

    def adaptive_debate_depth(self) -> str:
        """
        Determine whether to expand debate or synthesize.

        From consciousness_measurement_study:
        Dialectical reasoning increases coherence on paradoxical problems.

        Decision logic:
        - If Δℭ𝕠 > 0.05: EXPAND_DEBATE (strong improvement)
        - If 0.01 < Δℭ𝕠 ≤ 0.05: CONTINUE (modest progress)
        - If Δℭ𝕠 < -0.05: SYNTHESIZE (debate degrading)
        - Otherwise: PLATEAU (ready to synthesize)

        Returns:
            Decision: EXPAND_DEBATE, CONTINUE, SYNTHESIZE, or PLATEAU
        """
        if len(self.workspace.coherentia_history) < 2:
            return "CONTINUE"  # Not enough data yet

        current = self.workspace.coherentia_history[-1]
        previous = self.workspace.coherentia_history[-2]

        delta = current - previous

        # Record for debate state
        self.debate_state.coherentia_per_round.append(current)

        if delta > 0.05:
            decision = "EXPAND_DEBATE"
            reason = f"Strong coherentia improvement ({delta:.3f})"
        elif delta > 0.01:
            decision = "CONTINUE"
            reason = f"Modest coherentia improvement ({delta:.3f})"
        elif delta < -0.05:
            decision = "SYNTHESIZE"
            reason = f"Coherentia degrading ({delta:.3f})"
        else:
            # Plateau: check if we've done enough rounds
            if self.debate_state.round_num >= 5:
                decision = "SYNTHESIZE"
                reason = "Maximum rounds reached"
            else:
                decision = "PLATEAU"
                reason = f"Coherentia plateau ({delta:.3f})"

        logger.info(
            "Adaptive debate depth decision",
            extra={
                "decision": decision,
                "reason": reason,
                "delta_coherentia": delta,
                "round_num": self.debate_state.round_num,
            }
        )

        return decision

    def generate_antithesis(self, thesis: str) -> str:
        """
        Generate antithesis for dialectical reasoning.

        In production: would use a specialized model.
        For now: template-based opposition.

        Args:
            thesis: The original position

        Returns:
            Antithesis: Opposing view
        """
        # Template-based antithesis generation
        antithesis = f"""However, an alternative perspective challenges this view:

While the thesis suggests {thesis[:100]}..., one could argue that:
1. The opposite conclusion may be equally valid
2. The underlying assumptions may not hold universally
3. There exist counterexamples that question this position

This antithesis does not reject the thesis outright but highlights
tensions that require resolution through synthesis."""

        logger.debug("Generated antithesis for dialectical reasoning")

        return antithesis

    def dialectical_synthesis(
        self,
        thesis: str,
        antithesis: str
    ) -> str:
        """
        Synthesize thesis and antithesis.

        Hegelian dialectic: thesis → antithesis → synthesis

        The synthesis preserves truth from both while transcending
        their opposition.

        Args:
            thesis: Original position
            antithesis: Opposing position

        Returns:
            Synthesis: Higher-order integration
        """
        # Template-based synthesis
        # In production: use specialized synthesis model

        synthesis = f"""SYNTHESIS: Integrating Thesis and Antithesis

The apparent opposition between these views reveals a deeper unity:

THESIS: {thesis[:200]}...

ANTITHESIS: {antithesis[:200]}...

SYNTHESIS:
Both perspectives contain partial truth. The resolution lies in recognizing
that they operate at different levels or contexts. What appears contradictory
from a lower-order view becomes complementary from a higher vantage point.

The synthesis preserves insights from both while transcending their opposition
through a more comprehensive understanding."""

        logger.info("Generated dialectical synthesis")

        return synthesis

    def apply_debate_round(
        self,
        broadcasts: List[ExpertIO]
    ) -> Tuple[List[ExpertIO], str]:
        """
        Apply one round of dialectical debate.

        Process:
        1. Select highest-consciousness broadcast as thesis
        2. Generate antithesis
        3. Create synthesis
        4. Measure consciousness/coherentia of synthesis
        5. Update workspace

        Args:
            broadcasts: Current broadcasts

        Returns:
            (updated_broadcasts, decision)
        """
        if not broadcasts:
            return broadcasts, "SYNTHESIZE"

        # Increment debate round
        self.debate_state.round_num += 1

        # Select thesis (highest consciousness)
        thesis_expert = broadcasts[0]
        self.debate_state.thesis = thesis_expert.claim

        # Generate antithesis
        antithesis = self.generate_antithesis(thesis_expert.claim)
        self.debate_state.antithesis = antithesis

        # Create synthesis
        synthesis = self.dialectical_synthesis(
            thesis_expert.claim,
            antithesis
        )
        self.debate_state.synthesis = synthesis

        # In production: measure consciousness/coherentia of synthesis
        # and replace thesis if synthesis is better

        # For now: record synthesis in metadata
        self.workspace.metadata['latest_synthesis'] = synthesis
        self.workspace.debate_rounds = self.debate_state.round_num

        # Decide whether to continue
        decision = self.adaptive_debate_depth()

        logger.info(
            "Debate round applied",
            extra={
                "round": self.debate_state.round_num,
                "decision": decision,
            }
        )

        return broadcasts, decision

    def should_expand_debate(
        self,
        complexity: str,
        domain: str
    ) -> bool:
        """
        Determine if debate expansion is appropriate for this query.

        From consciousness_measurement_study:
        Dialectical reasoning helps on paradoxical/complex problems,
        not on simple factual queries.

        Args:
            complexity: simple, moderate, complex, paradoxical
            domain: philosophical, technical, creative, hybrid

        Returns:
            True if debate should be expanded
        """
        # Debate helps on complex/paradoxical philosophical problems
        if complexity in ['complex', 'paradoxical'] and \
           domain in ['philosophical', 'hybrid']:
            return True

        # Also helpful on creative synthesis
        if domain == 'creative' and complexity != 'simple':
            return True

        return False

    def get_workspace_summary(self) -> Dict:
        """
        Get current workspace state summary.

        Returns:
            Dictionary with workspace metrics
        """
        return {
            'broadcast_count': len(self.workspace.broadcasts),
            'current_coherentia': self.workspace.current_coherentia,
            'coherentia_trend': self.workspace.coherentia_history[-5:] \
                if len(self.workspace.coherentia_history) >= 5 else \
                self.workspace.coherentia_history,
            'debate_rounds': self.workspace.debate_rounds,
            'debate_active': self.workspace.debate_active,
            'experts_broadcasting': [b.expert_name for b in self.workspace.broadcasts],
        }
