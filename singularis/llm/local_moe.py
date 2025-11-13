"""
Local Mixture of Experts using LM Studio models.

Architecture:
- 4x Qwen3-VL-8B experts (vision + reasoning)
- 1x Phi-4 synthesizer (final decision)

All models run in parallel via LM Studio.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .lmstudio_client import LMStudioClient, LMStudioConfig


@dataclass
class LocalMoEConfig:
    """Configuration for local MoE system."""
    num_experts: int = 4  # Number of Qwen3 experts
    expert_model: str = "qwen/qwen3-4b-thinking-2507"
    synthesizer_model: str = "microsoft/phi-4"
    base_url: str = "http://localhost:1234/v1"
    timeout: int = 15  # Timeout per expert query (increased for thinking models)
    max_tokens: int = 1024  # Increased for better reasoning depth


class LocalMoEOrchestrator:
    """
    Local Mixture of Experts orchestrator.
    
    Uses 4 Qwen3-VL experts for parallel analysis, then Phi-4 to synthesize
    the final decision. All models run locally via LM Studio.
    """
    
    def __init__(self, config: LocalMoEConfig):
        """Initialize local MoE system."""
        self.config = config
        
        # Expert specializations
        self.expert_roles = [
            "visual_perception",      # Focus on what's visible
            "spatial_reasoning",      # Focus on positioning and movement
            "threat_assessment",      # Focus on danger and combat
            "opportunity_detection"   # Focus on items, NPCs, interactions
        ]
        
        # Initialize expert clients
        self.experts: List[LMStudioClient] = []
        self.synthesizer: Optional[LMStudioClient] = None
        
        logger.info(f"Local MoE initialized: {config.num_experts} experts + 1 synthesizer")
    
    async def initialize(self):
        """Initialize all LM Studio clients."""
        logger.info("Initializing local MoE experts...")
        
        # Model names with instance suffixes (as shown in LM Studio)
        expert_models = [
            "qwen/qwen3-4b-thinking-2507",      # Instance 1
            "microsoft/phi-4-mini-reasoning",    # Instance 2
            "microsoft/phi-4-mini-reasoning:2",  # Instance 3
            "microsoft/phi-4-mini-reasoning:3"   # Instance 4
        ]
        
        # Initialize experts with different model instances
        for i, role in enumerate(self.expert_roles[:self.config.num_experts]):
            expert_config = LMStudioConfig(
                base_url=self.config.base_url,
                model_name=expert_models[i],  # Use specific instance
                timeout=self.config.timeout,
                max_tokens=self.config.max_tokens
            )
            expert = LMStudioClient(expert_config)
            self.experts.append(expert)
            logger.info(f"✓ Expert {i+1}/{self.config.num_experts}: {expert_models[i]} ({role})")
        
        # Initialize synthesizer
        synth_config = LMStudioConfig(
            base_url=self.config.base_url,
            model_name=self.config.synthesizer_model,
            timeout=self.config.timeout,
            max_tokens=512  # Increased for better synthesis
        )
        self.synthesizer = LMStudioClient(synth_config)
        logger.info(f"✓ Synthesizer: {self.config.synthesizer_model}")
        
        logger.info("Local MoE initialization complete - all models can run in parallel!")
    
    async def _query_expert(
        self,
        expert_idx: int,
        role: str,
        prompt: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Query a single expert."""
        try:
            expert = self.experts[expert_idx]
            
            # Add role-specific context to system prompt
            role_context = f"\nYour role: {role}. Focus on this aspect in your analysis."
            full_system = system_prompt + role_context
            
            response = await expert.generate(
                prompt=prompt,
                system_prompt=full_system,
                max_tokens=self.config.max_tokens
            )
            
            return {
                'expert': expert_idx,
                'role': role,
                'response': response.get('content', ''),
                'success': True
            }
        except Exception as e:
            logger.warning(f"Expert {expert_idx} ({role}) failed: {e}")
            return {
                'expert': expert_idx,
                'role': role,
                'response': '',
                'success': False
            }
    
    async def get_action_recommendation(
        self,
        perception: Dict[str, Any],
        game_state: Any,
        available_actions: List[str],
        q_values: Dict[str, float],
        motivation: str = "exploration"
    ) -> Optional[Tuple[str, str]]:
        """
        Get action recommendation from MoE.
        
        Returns:
            Tuple of (action, reasoning) or None if failed
        """
        logger.info("[LOCAL-MOE] Starting parallel expert queries...")
        
        # Build prompt for experts
        visual_info = perception.get('visual_analysis', 'No visual analysis available')
        scene_type = perception.get('scene_type', 'unknown')
        
        expert_prompt = f"""Analyze this Skyrim gameplay situation:

SCENE: {scene_type}
VISUAL: {visual_info[:200]}...

GAME STATE:
- Health: {game_state.health:.0f}%
- Combat: {game_state.in_combat}
- Enemies: {game_state.enemies_nearby}

AVAILABLE ACTIONS: {', '.join(available_actions)}

TOP Q-VALUES:
{', '.join([f"{a}={v:.2f}" for a, v in sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:5]])}

Based on your role, recommend ONE action and explain why."""

        system_prompt = f"""You are an expert Skyrim AI assistant. 
Analyze the situation and recommend the best action.
Motivation: {motivation}
Be concise and decisive."""
        
        # Query all experts in parallel
        expert_tasks = [
            self._query_expert(i, role, expert_prompt, system_prompt)
            for i, role in enumerate(self.expert_roles[:self.config.num_experts])
        ]
        
        expert_responses = await asyncio.gather(*expert_tasks, return_exceptions=True)
        
        # Filter successful responses
        successful = [r for r in expert_responses if isinstance(r, dict) and r.get('success')]
        
        if not successful:
            logger.warning("[LOCAL-MOE] All experts failed")
            return None
        
        logger.info(f"[LOCAL-MOE] {len(successful)}/{self.config.num_experts} experts responded")
        
        # Synthesize with Phi-4
        synthesis_prompt = f"""You are synthesizing recommendations from {len(successful)} expert AI systems for Skyrim gameplay.

AVAILABLE ACTIONS: {', '.join(available_actions)}

EXPERT RECOMMENDATIONS:
"""
        for resp in successful:
            synthesis_prompt += f"\n{resp['role'].upper()}:\n{resp['response'][:300]}\n"
        
        synthesis_prompt += f"""

Synthesize these expert opinions into ONE final action recommendation.
Choose from: {', '.join(available_actions)}

Output format:
ACTION: <action_name>
REASONING: <brief explanation>"""
        
        try:
            logger.info("[LOCAL-MOE] Synthesizing with Phi-4...")
            synth_response = await self.synthesizer.generate(
                prompt=synthesis_prompt,
                system_prompt="You are a decisive AI synthesizer. Choose the best action based on expert consensus.",
                max_tokens=256
            )
            
            synth_text = synth_response.get('content', '')
            
            # Parse action from synthesis
            action = None
            reasoning = synth_text
            
            # Try to extract ACTION: line
            for line in synth_text.split('\n'):
                if line.strip().startswith('ACTION:'):
                    action_text = line.split('ACTION:')[1].strip()
                    # Find matching action
                    for avail_action in available_actions:
                        if avail_action.lower() in action_text.lower():
                            action = avail_action
                            break
                    break
            
            # Fallback: find any available action in text
            if not action:
                for avail_action in available_actions:
                    if avail_action in synth_text.lower():
                        action = avail_action
                        break
            
            if action:
                logger.info(f"[LOCAL-MOE] ✓ Final decision: {action}")
                return (action, reasoning[:200])
            else:
                logger.warning("[LOCAL-MOE] Could not extract action from synthesis")
                return None
                
        except Exception as e:
            logger.error(f"[LOCAL-MOE] Synthesis failed: {e}")
            return None
