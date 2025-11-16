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
from .response_cache import LLMResponseCache


@dataclass
class LocalMoEConfig:
    """Configuration for local MoE system."""
    num_experts: int = 4  # Number of phi-4-mini-reasoning experts
    expert_model: str = "microsoft/phi-4-mini-reasoning"
    synthesizer_model: str = "microsoft/phi-4-mini-reasoning"  # Use phi-4-mini for synthesis
    fallback_synthesizer: str = "mistralai/mistral-nemo-instruct-2407"  # Fallback if phi-4 fails
    base_url: str = "http://localhost:1234/v1"
    timeout: int = 20  # Increased timeout per expert query for reliability
    synthesis_timeout: int = 15  # Separate timeout for synthesis step
    max_tokens: int = 1024


class LocalMoEOrchestrator:
    """
    Orchestrates a local Mixture of Experts (MoE) system, using multiple
    language models running in parallel via LM Studio to analyze a situation
    and recommend an action.
    """
    
    def __init__(self, config: LocalMoEConfig):
        """
        Initializes the LocalMoEOrchestrator.

        Args:
            config (LocalMoEConfig): A configuration object for the MoE system.
        """
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
        self.fallback_synthesizer: Optional[LMStudioClient] = None
        
        # Initialize response cache
        self.cache = LLMResponseCache(
            max_size=200,
            ttl_seconds=120.0,  # Cache for 2 minutes
            enable_similarity=True
        )
        
        logger.info(f"Local MoE initialized: {config.num_experts} experts + 1 synthesizer + cache")
    
    async def initialize(self):
        """Initializes the LM Studio clients for the experts and synthesizers."""
        logger.info("Initializing local MoE experts...")
        
        # Model names matching all loaded instances in LM Studio
        # Using all available phi-4-mini-reasoning instances for true parallel MoE
        expert_models = [
            "microsoft/phi-4-mini-reasoning",    # Instance 1
            "microsoft/phi-4-mini-reasoning:2",  # Instance 2
            "microsoft/phi-4-mini-reasoning:3",  # Instance 3
            "microsoft/phi-4-mini-reasoning:4"   # Instance 4
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
            timeout=self.config.synthesis_timeout,  # Use dedicated synthesis timeout
            max_tokens=512  # Increased for better synthesis
        )
        self.synthesizer = LMStudioClient(synth_config)
        logger.info(f"✓ Synthesizer: {self.config.synthesizer_model}")
        
        # Initialize fallback synthesizer (Mistral-Nemo)
        if self.config.fallback_synthesizer:
            fallback_config = LMStudioConfig(
                base_url=self.config.base_url,
                model_name=self.config.fallback_synthesizer,
                timeout=self.config.synthesis_timeout,
                max_tokens=512
            )
            self.fallback_synthesizer = LMStudioClient(fallback_config)
            logger.info(f"✓ Fallback synthesizer: {self.config.fallback_synthesizer}")
        else:
            self.fallback_synthesizer = None
        
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
    
    async def close(self):
        """Closes the aiohttp sessions for all expert and synthesizer clients."""
        for expert in self.experts:
            if expert.session:
                await expert.session.close()
        
        if self.synthesizer and self.synthesizer.session:
            await self.synthesizer.session.close()
        
        logger.info("Local MoE closed - all sessions terminated")
    
    async def get_action_recommendation(
        self,
        perception: Dict[str, Any],
        game_state: Any,
        available_actions: List[str],
        q_values: Dict[str, float],
        motivation: str = "exploration"
    ) -> Optional[Tuple[str, str]]:
        """
        Gets an action recommendation from the MoE system.

        This method queries the expert models in parallel, synthesizes their
        responses, and returns a single action and the reasoning behind it.
        It also uses a cache to speed up responses for similar situations.

        Args:
            perception (Dict[str, Any]): A dictionary of perception data.
            game_state (Any): The current game state.
            available_actions (List[str]): A list of available actions.
            q_values (Dict[str, float]): A dictionary of Q-values for the actions.
            motivation (str, optional): The current motivation of the AGI.
                                      Defaults to "exploration".

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the recommended action
                                      and the reasoning, or None if the process
                                      fails.
        """
        # Check cache first
        scene_type = perception.get('scene_type', 'unknown')
        # Convert SceneType enum to string if needed
        if hasattr(scene_type, 'value'):
            scene_type = scene_type.value
        elif not isinstance(scene_type, str):
            scene_type = str(scene_type)
        
        health = getattr(game_state, 'health', 100.0)
        in_combat = getattr(game_state, 'in_combat', False)
        actions_tuple = tuple(sorted(available_actions))
        
        cached = self.cache.get(
            scene_type=scene_type,
            health=health,
            in_combat=in_combat,
            available_actions=actions_tuple
        )
        
        if cached is not None:
            logger.info(f"[LOCAL-MOE] ✓ Cache hit: {cached[0]}")
            return cached
        
        logger.info("[LOCAL-MOE] Cache miss - starting parallel expert queries...")
        
        # Build prompt for experts
        visual_info = perception.get('visual_analysis', 'No visual analysis available')
        scene_type_str = perception.get('scene_type', 'unknown')
        
        # Convert scene_type to string if it's an enum
        if hasattr(scene_type_str, 'value'):
            scene_type_str = scene_type_str.value
        elif not isinstance(scene_type_str, str):
            scene_type_str = str(scene_type_str)
        
        expert_prompt = f"""Analyze this Skyrim gameplay situation:

SCENE: {scene_type_str}
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
                
                # Cache the result
                result = (action, reasoning[:200])
                self.cache.put(
                    scene_type=scene_type,
                    health=health,
                    in_combat=in_combat,
                    available_actions=actions_tuple,
                    response=result
                )
                
                return result
            else:
                logger.warning("[LOCAL-MOE] Could not extract action from synthesis")
                return None
                
        except Exception as e:
            logger.error(f"[LOCAL-MOE] Synthesis failed: {e}")
            
            # Try fallback synthesizer (Mistral-Nemo)
            if self.fallback_synthesizer:
                try:
                    logger.info("[LOCAL-MOE] Retrying synthesis with Mistral-Nemo fallback...")
                    synth_response = await self.fallback_synthesizer.generate(
                        prompt=synthesis_prompt,
                        system_prompt="You are a decisive AI synthesizer. Choose the best action based on expert consensus.",
                        max_tokens=256
                    )
                    
                    synth_text = synth_response.get('content', '')
                    
                    # Parse action
                    action = None
                    reasoning = synth_text
                    
                    for line in synth_text.split('\n'):
                        if line.strip().startswith('ACTION:'):
                            action_text = line.split('ACTION:')[1].strip()
                            for avail_action in available_actions:
                                if avail_action.lower() in action_text.lower():
                                    action = avail_action
                                    break
                            break
                    
                    if not action:
                        for avail_action in available_actions:
                            if avail_action in synth_text.lower():
                                action = avail_action
                                break
                    
                    if action:
                        logger.info(f"[LOCAL-MOE] ✓ Fallback synthesis succeeded: {action}")
                        
                        # Cache the fallback result
                        result = (action, reasoning[:200])
                        self.cache.put(
                            scene_type=scene_type,
                            health=health,
                            in_combat=in_combat,
                            available_actions=actions_tuple,
                            response=result
                        )
                        
                        return result
                    else:
                        logger.warning("[LOCAL-MOE] Fallback synthesis could not extract action")
                        return None
                        
                except Exception as fallback_err:
                    logger.error(f"[LOCAL-MOE] Fallback synthesis also failed: {fallback_err}")
                    return None
            else:
                # No fallback synthesizer available
                logger.warning("[LOCAL-MOE] No fallback synthesizer configured")
                return None
