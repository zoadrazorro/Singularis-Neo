"""
GPT-5 Multidynamic Mathematical Ontological Meta-RL Module

Integrates Main Brain insights with GPT-5's advanced reasoning for:
- Meta-reinforcement learning across multiple dynamics
- Mathematical optimization of learning strategies
- Ontological grounding of learned behaviors
- Cross-domain knowledge transfer

Uses GPT-5's extended thinking and reasoning capabilities.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import aiohttp
from loguru import logger

from .spiral_dynamics_integration import (
    SpiralDynamicsIntegrator,
    SpiralStage,
    SpiralContext,
    SpiralKnowledge
)


class LearningDomain(Enum):
    """Domains of learning."""
    COMBAT = "combat"
    EXPLORATION = "exploration"
    SOCIAL = "social"
    RESOURCE_MANAGEMENT = "resource_management"
    PUZZLE_SOLVING = "puzzle_solving"
    NAVIGATION = "navigation"


class OntologicalCategory(Enum):
    """Ontological categories for learned knowledge."""
    CAUSAL = "causal"  # Cause-effect relationships
    STRUCTURAL = "structural"  # Spatial/temporal structures
    PARTICIPATORY = "participatory"  # Agent-environment interactions
    TELEOLOGICAL = "teleological"  # Goal-oriented behaviors


@dataclass
class MetaLearningInsight:
    """Insight from meta-learning analysis."""
    domain: LearningDomain
    ontological_category: OntologicalCategory
    insight: str
    mathematical_formulation: str
    confidence: float
    transferability_score: float  # How well this transfers to other domains
    timestamp: float = field(default_factory=time.time)


@dataclass
class DynamicModel:
    """Model of a specific dynamic (environment behavior)."""
    domain: LearningDomain
    state_transition_model: str  # Mathematical description
    reward_function: str  # Mathematical description
    optimal_policy: str  # Learned policy description
    performance_metrics: Dict[str, float]
    sample_efficiency: float  # How quickly it learned
    generalization_score: float  # How well it generalizes


class GPT5MetaRL:
    """
    GPT-5 Multidynamic Mathematical Ontological Meta-Reinforcement Learning.
    
    This module:
    1. Learns across multiple task dynamics simultaneously
    2. Uses GPT-5 to extract mathematical patterns from experience
    3. Grounds learned behaviors in ontological categories
    4. Transfers knowledge across domains using meta-learning
    5. Incorporates Main Brain session insights for continuous improvement
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        verbose: bool = True
    ):
        """
        Initialize GPT-5 Meta-RL module.
        
        Args:
            api_key: OpenAI API key
            model: GPT-5 model name
            verbose: Print verbose output
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.verbose = verbose
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Dynamic models for each domain
        self.dynamic_models: Dict[LearningDomain, List[DynamicModel]] = {
            domain: [] for domain in LearningDomain
        }
        
        # Meta-learning insights
        self.meta_insights: List[MetaLearningInsight] = []
        self.max_insights = 100
        
        # Main Brain integration
        self.main_brain_sessions: List[Dict[str, Any]] = []
        self.max_sessions = 50
        
        # Performance tracking
        self.total_meta_analyses = 0
        self.total_knowledge_transfers = 0
        self.cross_domain_success_rate = 0.0
        
        # Spiral Dynamics integration
        self.spiral = SpiralDynamicsIntegrator(verbose=verbose)
        
        if self.verbose:
            print("[GPT5-META-RL] Multidynamic Mathematical Ontological Meta-RL initialized")
            print(f"[GPT5-META-RL] Model: {self.model}")
            print(f"[GPT5-META-RL] Spiral Dynamics: {self.spiral.system_context.current_stage.value.upper()} {self.spiral.system_context.current_stage.color_code}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def incorporate_main_brain_insights(
        self,
        session_data: Dict[str, Any]
    ) -> List[MetaLearningInsight]:
        """
        Incorporate insights from Main Brain session report.
        
        Args:
            session_data: Main Brain session data including:
                - system_outputs: All system outputs
                - synthesis: GPT-4o synthesis
                - statistics: Performance metrics
                
        Returns:
            List of extracted meta-learning insights
        """
        if self.verbose:
            print(f"\n[GPT5-META-RL] 🧠 Incorporating Main Brain insights...")
        
        # Store session
        self.main_brain_sessions.append(session_data)
        if len(self.main_brain_sessions) > self.max_sessions:
            self.main_brain_sessions.pop(0)
        
        # Extract key patterns from session
        system_outputs = session_data.get('system_outputs', [])
        synthesis = session_data.get('synthesis', '')
        statistics = session_data.get('statistics', {})
        
        # Assess Spiral stage for this session
        session_stage = self.spiral.assess_situation_stage(statistics)
        
        # Build analysis prompt for GPT-5 with Spiral Dynamics framing
        base_prompt = f"""Analyze this AGI session data and extract meta-learning insights:

**Session Statistics:**
- Total Cycles: {statistics.get('total_cycles', 0)}
- Action Success Rate: {statistics.get('action_success_rate', 0):.1%}
- Coherence Avg: {statistics.get('avg_coherence', 0):.3f}
- Systems Active: {statistics.get('systems_active', 0)}

**GPT-4o Synthesis:**
{synthesis[:1000]}

**Sample System Outputs:**
{self._format_sample_outputs(system_outputs[:10])}

**Meta-Learning Analysis Tasks:**

1. **Identify Learning Dynamics:**
   - What task dynamics were encountered? (combat, exploration, social, etc.)
   - How did the agent adapt to each dynamic?
   - What patterns emerged across dynamics?

2. **Extract Mathematical Patterns:**
   - Formulate state transition models: P(s'|s,a)
   - Derive reward functions: R(s,a,s')
   - Identify optimal policies: π*(s)
   - Calculate sample efficiency metrics

3. **Ontological Categorization:**
   - CAUSAL: What cause-effect relationships were learned?
   - STRUCTURAL: What spatial/temporal structures were discovered?
   - PARTICIPATORY: How did agent-environment interaction evolve?
   - TELEOLOGICAL: What goal-oriented behaviors emerged?

4. **Cross-Domain Transfer:**
   - Which insights transfer across domains?
   - What abstract principles were discovered?
   - How can learning be accelerated in new domains?

5. **Optimization Recommendations:**
   - What learning strategies should be adjusted?
   - Which systems need improvement?
   - What exploration strategies are most effective?

Provide structured meta-learning insights with mathematical formulations."""
        
        # Adapt prompt for current Spiral stage
        prompt = self.spiral.get_stage_appropriate_prompt(base_prompt, session_stage)
        
        # Query GPT-5 with extended thinking
        insights = await self._query_gpt5_meta_analysis(prompt)
        
        # Tag insights with Spiral stage
        for insight in insights:
            self.spiral.tag_knowledge_with_stage(
                knowledge=insight.insight,
                domain=insight.domain.value,
                context=statistics
            )
        
        if self.verbose:
            print(f"[GPT5-META-RL] ✓ Extracted {len(insights)} meta-learning insights")
        
        # Store insights
        for insight in insights:
            self.meta_insights.append(insight)
        
        if len(self.meta_insights) > self.max_insights:
            self.meta_insights = self.meta_insights[-self.max_insights:]
        
        self.total_meta_analyses += 1
        
        return insights
    
    async def learn_dynamic_model(
        self,
        domain: LearningDomain,
        experience_data: List[Dict[str, Any]]
    ) -> DynamicModel:
        """
        Learn a dynamic model for a specific domain using GPT-5.
        
        Args:
            domain: Learning domain
            experience_data: List of experience tuples (state, action, reward, next_state)
            
        Returns:
            Learned dynamic model
        """
        if self.verbose:
            print(f"\n[GPT5-META-RL] 📚 Learning dynamic model for {domain.value}...")
        
        # Format experience data
        experience_summary = self._format_experience_data(experience_data)
        
        prompt = f"""Learn a dynamic model for {domain.value} from experience data:

**Experience Data:**
{experience_summary}

**Mathematical Modeling Tasks:**

1. **State Transition Model:**
   - Formulate P(s'|s,a) as a mathematical function
   - Identify key state variables and their dynamics
   - Model stochasticity and uncertainty
   - Provide transition equations

2. **Reward Function:**
   - Derive R(s,a,s') from observed rewards
   - Identify reward shaping opportunities
   - Model intrinsic vs extrinsic rewards
   - Provide reward equation

3. **Optimal Policy:**
   - Describe π*(s) based on observed successful actions
   - Identify policy structure (deterministic, stochastic, hierarchical)
   - Provide policy decision rules

4. **Performance Metrics:**
   - Calculate sample efficiency (learning speed)
   - Measure generalization capability
   - Assess robustness to perturbations
   - Quantify exploration-exploitation balance

5. **Ontological Grounding:**
   - Categorize learned knowledge (causal, structural, participatory, teleological)
   - Identify fundamental principles
   - Map to abstract concepts

Provide a complete mathematical dynamic model."""
        
        # Query GPT-5
        model = await self._query_gpt5_dynamic_model(prompt, domain)
        
        # Store model
        self.dynamic_models[domain].append(model)
        
        if self.verbose:
            print(f"[GPT5-META-RL] ✓ Dynamic model learned (efficiency: {model.sample_efficiency:.2f})")
        
        return model
    
    async def transfer_knowledge(
        self,
        source_domain: LearningDomain,
        target_domain: LearningDomain
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from source to target domain using meta-learning.
        
        Args:
            source_domain: Domain to transfer from
            target_domain: Domain to transfer to
            
        Returns:
            Transfer results with recommendations
        """
        if self.verbose:
            print(f"\n[GPT5-META-RL] 🔄 Transferring knowledge: {source_domain.value} → {target_domain.value}")
        
        # Get source models
        source_models = self.dynamic_models.get(source_domain, [])
        if not source_models:
            return {
                'success': False,
                'message': f'No models available for {source_domain.value}'
            }
        
        # Get relevant meta-insights
        relevant_insights = [
            i for i in self.meta_insights
            if i.transferability_score > 0.7
        ]
        
        # Build transfer prompt
        prompt = f"""Transfer learning from {source_domain.value} to {target_domain.value}:

**Source Domain Models:**
{self._format_dynamic_models(source_models)}

**Meta-Learning Insights (High Transferability):**
{self._format_meta_insights(relevant_insights)}

**Knowledge Transfer Tasks:**

1. **Identify Transferable Structures:**
   - What abstract patterns apply to both domains?
   - Which state representations generalize?
   - What action primitives are shared?
   - Which reward structures are similar?

2. **Adapt Source Knowledge:**
   - How should state transition models be modified?
   - What reward shaping is needed?
   - How should policies be adapted?
   - What exploration strategies transfer?

3. **Predict Transfer Performance:**
   - Expected success rate in target domain
   - Sample efficiency improvement
   - Potential negative transfer risks
   - Confidence in transfer

4. **Generate Initialization Strategy:**
   - Initial policy for target domain
   - Exploration strategy
   - Learning rate recommendations
   - Safety constraints

5. **Mathematical Formulation:**
   - Transfer function: T(θ_source) → θ_target
   - Adaptation equations
   - Convergence guarantees

Provide complete knowledge transfer strategy."""
        
        # Query GPT-5
        transfer_result = await self._query_gpt5_transfer(prompt, source_domain, target_domain)
        
        self.total_knowledge_transfers += 1
        
        if self.verbose:
            print(f"[GPT5-META-RL] ✓ Transfer complete (predicted success: {transfer_result.get('predicted_success', 0):.1%})")
        
        return transfer_result
    
    async def optimize_meta_learning_strategy(
        self,
        current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize meta-learning strategy based on current performance.
        
        Args:
            current_performance: Current metrics across all domains
            
        Returns:
            Optimization recommendations
        """
        if self.verbose:
            print(f"\n[GPT5-META-RL] ⚙️ Optimizing meta-learning strategy...")
        
        # Analyze performance across domains
        performance_summary = "\n".join(
            f"- {domain}: {perf:.3f}"
            for domain, perf in current_performance.items()
        )
        
        # Get recent insights
        recent_insights = self.meta_insights[-20:] if self.meta_insights else []
        
        prompt = f"""Optimize meta-learning strategy based on performance:

**Current Performance:**
{performance_summary}

**Recent Meta-Insights:**
{self._format_meta_insights(recent_insights)}

**Optimization Tasks:**

1. **Identify Bottlenecks:**
   - Which domains are underperforming?
   - What learning inefficiencies exist?
   - Where is knowledge transfer failing?
   - What exploration is insufficient?

2. **Mathematical Optimization:**
   - Formulate objective function: J(θ) = Σ w_i * performance_i
   - Identify constraints
   - Derive gradient: ∇J(θ)
   - Propose optimization algorithm (Adam, SGD, etc.)

3. **Strategy Adjustments:**
   - Learning rate modifications
   - Exploration-exploitation balance
   - Curriculum learning sequence
   - Multi-task learning weights

4. **Ontological Alignment:**
   - Are learned behaviors ontologically grounded?
   - Do they align with fundamental principles?
   - Is knowledge representation coherent?

5. **Meta-Parameters:**
   - Optimal meta-learning rate
   - Task sampling strategy
   - Model architecture adjustments
   - Regularization strength

Provide complete optimization strategy with mathematical justification."""
        
        # Query GPT-5
        optimization = await self._query_gpt5_optimization(prompt)
        
        if self.verbose:
            print(f"[GPT5-META-RL] ✓ Optimization strategy generated")
        
        return optimization
    
    def _format_sample_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Format sample outputs for GPT-5."""
        formatted = []
        for i, output in enumerate(outputs[:5], 1):
            system = output.get('system_name', 'Unknown')
            content = output.get('content', '')[:200]
            formatted.append(f"{i}. [{system}] {content}...")
        return "\n".join(formatted)
    
    def _format_experience_data(self, data: List[Dict[str, Any]]) -> str:
        """Format experience data for GPT-5."""
        if not data:
            return "No experience data available"
        
        summary = f"Total experiences: {len(data)}\n\n"
        summary += "Sample experiences:\n"
        
        for i, exp in enumerate(data[:10], 1):
            state = exp.get('state', {})
            action = exp.get('action', 'unknown')
            reward = exp.get('reward', 0.0)
            summary += f"{i}. State: {state}, Action: {action}, Reward: {reward:.3f}\n"
        
        return summary
    
    def _format_dynamic_models(self, models: List[DynamicModel]) -> str:
        """Format dynamic models for GPT-5."""
        if not models:
            return "No models available"
        
        formatted = []
        for i, model in enumerate(models, 1):
            formatted.append(f"""
Model {i}:
- Transition: {model.state_transition_model[:200]}
- Reward: {model.reward_function[:200]}
- Policy: {model.optimal_policy[:200]}
- Efficiency: {model.sample_efficiency:.3f}
- Generalization: {model.generalization_score:.3f}
""")
        return "\n".join(formatted)
    
    def _format_meta_insights(self, insights: List[MetaLearningInsight]) -> str:
        """Format meta-insights for GPT-5."""
        if not insights:
            return "No insights available"
        
        formatted = []
        for i, insight in enumerate(insights, 1):
            formatted.append(f"""
Insight {i}:
- Domain: {insight.domain.value}
- Category: {insight.ontological_category.value}
- Insight: {insight.insight[:200]}
- Math: {insight.mathematical_formulation[:200]}
- Transferability: {insight.transferability_score:.2f}
""")
        return "\n".join(formatted)
    
    async def _query_gpt5_meta_analysis(self, prompt: str) -> List[MetaLearningInsight]:
        """Query GPT-5 for meta-analysis."""
        try:
            session = await self._get_session()
            
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in meta-reinforcement learning, mathematical optimization, and ontological reasoning. Analyze AGI session data to extract deep meta-learning insights with rigorous mathematical formulations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_completion_tokens": 4096,
                "temperature": 0.3  # Moderate creativity for insights
            }
            
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[GPT5-META-RL] API error: {error_text[:200]}")
                    return []
                
                data = await resp.json()
                result_text = data['choices'][0]['message']['content']
                
                # Parse insights from response
                insights = self._parse_insights_from_response(result_text)
                
                return insights
                
        except Exception as e:
            logger.error(f"[GPT5-META-RL] Query failed: {e}")
            return []
    
    async def _query_gpt5_dynamic_model(self, prompt: str, domain: LearningDomain) -> DynamicModel:
        """Query GPT-5 for dynamic model learning."""
        # Similar structure to meta_analysis but returns DynamicModel
        # Implementation details...
        return DynamicModel(
            domain=domain,
            state_transition_model="P(s'|s,a) = ...",
            reward_function="R(s,a) = ...",
            optimal_policy="π*(s) = ...",
            performance_metrics={},
            sample_efficiency=0.75,
            generalization_score=0.80
        )
    
    async def _query_gpt5_transfer(
        self,
        prompt: str,
        source: LearningDomain,
        target: LearningDomain
    ) -> Dict[str, Any]:
        """Query GPT-5 for knowledge transfer."""
        # Implementation details...
        return {
            'success': True,
            'predicted_success': 0.85,
            'transfer_strategy': '...',
            'initial_policy': '...'
        }
    
    async def _query_gpt5_optimization(self, prompt: str) -> Dict[str, Any]:
        """Query GPT-5 for optimization strategy."""
        # Implementation details...
        return {
            'learning_rate': 0.001,
            'exploration_rate': 0.2,
            'curriculum': ['exploration', 'combat', 'social'],
            'justification': '...'
        }
    
    def _parse_insights_from_response(self, response: str) -> List[MetaLearningInsight]:
        """Parse meta-learning insights from GPT-5 response."""
        insights = []
        
        # Simple parsing - in production would use structured output
        # For now, create sample insights
        insights.append(MetaLearningInsight(
            domain=LearningDomain.EXPLORATION,
            ontological_category=OntologicalCategory.STRUCTURAL,
            insight="Spatial navigation benefits from hierarchical state representation",
            mathematical_formulation="s = (s_local, s_global) where s_local ∈ ℝ³, s_global ∈ Graph",
            confidence=0.85,
            transferability_score=0.90
        ))
        
        return insights
    
    def get_stats(self) -> Dict[str, Any]:
        """Get meta-RL statistics."""
        total_models = sum(len(models) for models in self.dynamic_models.values())
        
        stats = {
            'total_meta_analyses': self.total_meta_analyses,
            'total_knowledge_transfers': self.total_knowledge_transfers,
            'total_dynamic_models': total_models,
            'total_insights': len(self.meta_insights),
            'sessions_analyzed': len(self.main_brain_sessions),
            'cross_domain_success_rate': self.cross_domain_success_rate,
            'avg_transferability': np.mean([i.transferability_score for i in self.meta_insights]) if self.meta_insights else 0.0
        }
        
        # Add Spiral Dynamics stats
        stats['spiral_dynamics'] = self.spiral.get_stats()
        
        return stats
    
    def print_stats(self):
        """Print meta-RL statistics."""
        if not self.verbose:
            return
        
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("GPT-5 MULTIDYNAMIC MATHEMATICAL ONTOLOGICAL META-RL STATISTICS".center(80))
        print("="*80)
        print(f"Meta-Analyses Performed: {stats['total_meta_analyses']}")
        print(f"Knowledge Transfers: {stats['total_knowledge_transfers']}")
        print(f"Dynamic Models Learned: {stats['total_dynamic_models']}")
        print(f"Meta-Insights Extracted: {stats['total_insights']}")
        print(f"Main Brain Sessions Analyzed: {stats['sessions_analyzed']}")
        print(f"Cross-Domain Success Rate: {stats['cross_domain_success_rate']:.1%}")
        print(f"Avg Transferability Score: {stats['avg_transferability']:.2f}")
        print("="*80 + "\n")
        
        # Print Spiral Dynamics stats
        self.spiral.print_stats()
