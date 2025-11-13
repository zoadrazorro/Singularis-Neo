"""
Lumen Integration System

Maps AGI subsystems to Lumen aspects from Metaluminosity framework:
1. Lumen Onticum (Being/Energy) - Drives and motivation
2. Lumen Structurale (Form/Information) - Structure and patterns
3. Lumen Participatum (Consciousness/Awareness) - Integration and awareness

Ensures balanced expression across all three aspects of Being.
"""

import numpy as np
from typing import Dict, Any, List, Set
from dataclasses import dataclass
from loguru import logger


@dataclass
class LumenBalance:
    """Balance across three Lumen aspects."""
    onticum: float  # Being/Energy
    structurale: float  # Form/Information
    participatum: float  # Consciousness/Awareness
    balance_score: float  # Overall balance (1.0 = perfect balance)
    imbalance_direction: str  # Which aspect is over/under-represented


class LumenIntegratedSystem:
    """
    Map subsystems to Lumen aspects.
    
    Provides philosophical grounding for the AGI architecture
    based on Metaluminosity framework.
    """
    
    def __init__(self):
        """Initialize Lumen integration."""
        
        # Lumen Onticum (Being/Energy)
        # These systems provide drive, motivation, and energy
        self.onticum_systems: Set[str] = {
            'emotion',  # Affective drives and energy
            'emotion_system',
            'motivation',  # Intrinsic motivation
            'hebbian',  # Learning energy and plasticity
            'spiritual',  # Contemplative energy
            'spiritual_awareness',
            'rl_system',  # Reinforcement drives
            'voice_system',  # Expressive energy
        }
        
        # Lumen Structurale (Form/Information)
        # These systems provide structure, form, and information
        self.structurale_systems: Set[str] = {
            'symbolic_logic',  # Formal logical structure
            'world_model',  # Information patterns and models
            'darwinian_logic',  # Modal structure and evolution
            'action_planning',  # Action structure
            'analytic_evolution',  # Analytical structure
            'perception',  # Sensory structure
            'sensorimotor',  # Spatial structure
            'video_interpreter',  # Visual information
        }
        
        # Lumen Participatum (Consciousness/Awareness)
        # These systems provide integration, awareness, and consciousness
        self.participatum_systems: Set[str] = {
            'consciousness',  # Direct consciousness integration
            'consciousness_bridge',
            'self_reflection',  # Meta-awareness
            'realtime_coordinator',  # Coordinated awareness
            'reward_tuning',  # Adaptive awareness
            'meta_strategist',  # Strategic awareness
            'strategic_planner',
        }
        
        # Statistics
        self.balance_history: List[LumenBalance] = []
        self.max_history = 100
        
        logger.info(
            f"[LUMEN] Initialized with {len(self.onticum_systems)} onticum, "
            f"{len(self.structurale_systems)} structurale, "
            f"{len(self.participatum_systems)} participatum systems"
        )
    
    def compute_lumen_balance(
        self,
        active_systems: Dict[str, Any],
        system_weights: Optional[Dict[str, float]] = None
    ) -> LumenBalance:
        """
        Measure balance across three Lumen aspects.
        
        Args:
            active_systems: Currently active systems with their outputs
            system_weights: Optional weights for each system
            
        Returns:
            Lumen balance measurements
        """
        if not active_systems:
            return LumenBalance(
                onticum=0.0,
                structurale=0.0,
                participatum=0.0,
                balance_score=0.0,
                imbalance_direction="none"
            )
        
        # Default weights
        if system_weights is None:
            system_weights = {s: 1.0 for s in active_systems.keys()}
        
        # Compute weighted activity for each Lumen
        onticum_activity = sum(
            system_weights.get(sys, 1.0)
            for sys in active_systems.keys()
            if sys in self.onticum_systems
        )
        
        structurale_activity = sum(
            system_weights.get(sys, 1.0)
            for sys in active_systems.keys()
            if sys in self.structurale_systems
        )
        
        participatum_activity = sum(
            system_weights.get(sys, 1.0)
            for sys in active_systems.keys()
            if sys in self.participatum_systems
        )
        
        # Normalize by number of systems in each category
        onticum_normalized = onticum_activity / len(self.onticum_systems) if self.onticum_systems else 0.0
        structurale_normalized = structurale_activity / len(self.structurale_systems) if self.structurale_systems else 0.0
        participatum_normalized = participatum_activity / len(self.participatum_systems) if self.participatum_systems else 0.0
        
        # Compute balance score (1.0 = perfect balance, 0.0 = complete imbalance)
        values = [onticum_normalized, structurale_normalized, participatum_normalized]
        
        if max(values) == 0:
            balance_score = 0.0
        else:
            # Use standard deviation to measure imbalance
            std_dev = np.std(values)
            mean_val = np.mean(values)
            
            # Normalize std dev by mean (coefficient of variation)
            if mean_val > 0:
                cv = std_dev / mean_val
                # Convert to balance score (0 CV = perfect balance)
                balance_score = 1.0 / (1.0 + cv)
            else:
                balance_score = 0.0
        
        # Determine imbalance direction
        imbalance_direction = self._determine_imbalance_direction(
            onticum_normalized,
            structurale_normalized,
            participatum_normalized
        )
        
        balance = LumenBalance(
            onticum=onticum_normalized,
            structurale=structurale_normalized,
            participatum=participatum_normalized,
            balance_score=balance_score,
            imbalance_direction=imbalance_direction
        )
        
        # Track history
        self.balance_history.append(balance)
        if len(self.balance_history) > self.max_history:
            self.balance_history.pop(0)
        
        logger.debug(
            f"[LUMEN] Balance: onticum={onticum_normalized:.2f}, "
            f"structurale={structurale_normalized:.2f}, "
            f"participatum={participatum_normalized:.2f}, "
            f"score={balance_score:.2f} ({imbalance_direction})"
        )
        
        return balance
    
    def _determine_imbalance_direction(
        self,
        onticum: float,
        structurale: float,
        participatum: float
    ) -> str:
        """Determine which Lumen is over/under-represented."""
        values = {
            'onticum': onticum,
            'structurale': structurale,
            'participatum': participatum
        }
        
        if max(values.values()) == 0:
            return "none"
        
        # Find highest and lowest
        max_lumen = max(values, key=values.get)
        min_lumen = min(values, key=values.get)
        
        max_val = values[max_lumen]
        min_val = values[min_lumen]
        
        # Check if imbalance is significant
        if max_val - min_val < 0.2:
            return "balanced"
        
        if max_val > min_val * 1.5:
            return f"{max_lumen}_excess"
        
        return f"{min_lumen}_deficit"
    
    def get_recommendations(
        self,
        balance: LumenBalance
    ) -> List[str]:
        """
        Get recommendations for improving Lumen balance.
        
        Args:
            balance: Current Lumen balance
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if balance.balance_score > 0.8:
            recommendations.append("Lumen balance is excellent. Continue current operation.")
            return recommendations
        
        # Analyze imbalance
        if "onticum_excess" in balance.imbalance_direction:
            recommendations.append(
                "Onticum (energy/drive) is over-represented. "
                "Consider activating more structural (world model, logic) or "
                "awareness (consciousness, reflection) systems."
            )
        elif "onticum_deficit" in balance.imbalance_direction:
            recommendations.append(
                "Onticum (energy/drive) is under-represented. "
                "Consider activating emotion, motivation, or spiritual systems "
                "to increase drive and energy."
            )
        
        if "structurale_excess" in balance.imbalance_direction:
            recommendations.append(
                "Structurale (form/information) is over-represented. "
                "Consider integrating more experiential (emotion, motivation) or "
                "conscious (reflection, awareness) processing."
            )
        elif "structurale_deficit" in balance.imbalance_direction:
            recommendations.append(
                "Structurale (form/information) is under-represented. "
                "Consider activating world model, symbolic logic, or "
                "analytical systems for better structure."
            )
        
        if "participatum_excess" in balance.imbalance_direction:
            recommendations.append(
                "Participatum (consciousness/awareness) is over-represented. "
                "Consider grounding in more concrete systems (perception, action) "
                "or structural systems (logic, models)."
            )
        elif "participatum_deficit" in balance.imbalance_direction:
            recommendations.append(
                "Participatum (consciousness/awareness) is under-represented. "
                "Consider activating consciousness bridge, self-reflection, or "
                "meta-strategic systems for higher-order awareness."
            )
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Lumen balance statistics."""
        if not self.balance_history:
            return {
                'balance_history_size': 0,
                'avg_balance_score': 0.0,
            }
        
        return {
            'balance_history_size': len(self.balance_history),
            'avg_balance_score': np.mean([b.balance_score for b in self.balance_history]),
            'avg_onticum': np.mean([b.onticum for b in self.balance_history]),
            'avg_structurale': np.mean([b.structurale for b in self.balance_history]),
            'avg_participatum': np.mean([b.participatum for b in self.balance_history]),
            'latest_balance': self.balance_history[-1].__dict__ if self.balance_history else None,
        }
    
    def classify_system(self, system_id: str) -> List[str]:
        """
        Classify which Lumen aspect(s) a system belongs to.
        
        Some systems may belong to multiple aspects.
        
        Args:
            system_id: System identifier
            
        Returns:
            List of Lumen aspects this system expresses
        """
        aspects = []
        
        if system_id in self.onticum_systems:
            aspects.append("onticum")
        
        if system_id in self.structurale_systems:
            aspects.append("structurale")
        
        if system_id in self.participatum_systems:
            aspects.append("participatum")
        
        return aspects if aspects else ["unknown"]
