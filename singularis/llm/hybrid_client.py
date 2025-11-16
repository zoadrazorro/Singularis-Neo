"""
Hybrid LLM Client - Orchestrates Gemini (vision) + Claude Sonnet 4 (reasoning) + Local LLMs (fallback)

Architecture:
- Gemini 2.0 Flash: Primary vision model for visual perception
- Claude Sonnet 4: Primary reasoning model for strategic thinking
- Local LLMs (optional): Fallback when cloud APIs are unavailable

All operations are async for maximum performance.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient, LMStudioConfig, ExpertLLMInterface


class TaskType(Enum):
    """Type of AI task to perform."""
    VISION = "vision"  # Visual perception and scene understanding
    REASONING = "reasoning"  # Strategic thinking and planning
    ACTION = "action"  # Fast tactical decisions
    WORLD_MODEL = "world_model"  # World understanding and causal reasoning


@dataclass
class HybridConfig:
    """Configuration for hybrid LLM system."""
    # Primary models (cloud)
    use_gemini_vision: bool = True
    gemini_model: str = "gemini-2.0-flash-exp"
    
    use_claude_reasoning: bool = True
    claude_model: str = "claude-sonnet-4-5-20250929"
    
    use_openai_world_model: bool = True
    openai_model: str = "gpt-5-2025-08-07"  # GPT-5 for world modeling
    
    # Fallback models (local - optional)
    use_local_fallback: bool = False
    local_base_url: str = "http://localhost:1234/v1"
    local_vision_model: str = "qwen/qwen3-vl-30b"  # Qwen3-VL for vision (MUST be vision-capable)
    local_reasoning_model: str = "mistralai/mistral-7b-instruct-v0.3"  # Mistral for reasoning
    local_action_model: str = "microsoft/phi-4"  # Phi-4 for fast action
    
    # Performance settings
    timeout: int = 90  # Increased for Claude reasoning and Gemini vision
    max_retries: int = 2
    fallback_on_error: bool = True
    
    # Rate limiting
    max_concurrent_requests: int = 4
    min_request_interval: float = 0.1


class HybridLLMClient:
    """
    A unified client that orchestrates multiple LLM providers, including
    Gemini for vision, Claude for reasoning, and local LLMs as a fallback.
    """
    
    def __init__(self, config: Optional[HybridConfig] = None):
        """
        Initializes the HybridLLMClient.

        Args:
            config (Optional[HybridConfig], optional): A `HybridConfig` object
                                                     containing the configuration for
                                                     the client. If not provided, a
                                                     default configuration is used.
                                                     Defaults to None.
        """
        self.config = config or HybridConfig()
        
        # Primary clients
        self.gemini: Optional[GeminiClient] = None
        self.claude: Optional[ClaudeClient] = None
        self.openai: Optional[OpenAIClient] = None
        
        # Fallback clients (optional)
        self.local_vision: Optional[ExpertLLMInterface] = None
        self.local_reasoning: Optional[ExpertLLMInterface] = None
        self.local_action: Optional[ExpertLLMInterface] = None
        
        # State tracking
        self.stats = {
            'gemini_calls': 0,
            'claude_calls': 0,
            'openai_calls': 0,
            'local_calls': 0,
            'fallback_activations': 0,
            'errors': 0,
            'total_time': 0.0
        }
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.last_request_time = 0.0
        
        logger.info("Hybrid LLM client initialized", extra={
            "gemini_enabled": self.config.use_gemini_vision,
            "claude_enabled": self.config.use_claude_reasoning,
            "openai_enabled": self.config.use_openai_world_model,
            "local_fallback": self.config.use_local_fallback
        })
    
    async def initialize(self):
        """Initializes all configured LLM clients."""
        logger.info("Initializing hybrid LLM system...")
        
        # Initialize Gemini (primary vision)
        if self.config.use_gemini_vision:
            try:
                self.gemini = GeminiClient(model=self.config.gemini_model)
                if self.gemini.is_available():
                    logger.info(f"✓ Gemini vision initialized: {self.config.gemini_model}")
                else:
                    logger.warning("Gemini API key not found (GEMINI_API_KEY)")
                    self.gemini = None
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.gemini = None
        
        # Initialize Claude (primary reasoning)
        if self.config.use_claude_reasoning:
            try:
                self.claude = ClaudeClient(model=self.config.claude_model)
                if self.claude.is_available():
                    logger.info(f"✓ Claude reasoning initialized: {self.config.claude_model}")
                else:
                    logger.warning("Claude API key not found (ANTHROPIC_API_KEY)")
                    self.claude = None
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
                self.claude = None
        
        # Initialize OpenAI (world modeling)
        if self.config.use_openai_world_model:
            try:
                self.openai = OpenAIClient(model=self.config.openai_model)
                if self.openai.is_available():
                    logger.info(f"✓ OpenAI world modeling initialized: {self.config.openai_model}")
                else:
                    logger.warning("OpenAI API key not found (OPENAI_API_KEY)")
                    self.openai = None
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.openai = None
        
        # Initialize local fallbacks (optional)
        if self.config.use_local_fallback:
            await self._initialize_local_fallbacks()
        
        # Report status
        self._report_initialization_status()
    
    async def _initialize_local_fallbacks(self):
        """Initialize local LLM fallbacks."""
        logger.info("Initializing local LLM fallbacks...")
        
        try:
            # Local vision model
            vision_config = LMStudioConfig(
                base_url=self.config.local_base_url,
                model_name=self.config.local_vision_model,
                temperature=0.5,
                max_tokens=1536
            )
            vision_client = LMStudioClient(vision_config)
            self.local_vision = ExpertLLMInterface(vision_client)
            logger.info(f"✓ Local vision fallback: {self.config.local_vision_model}")
            
            # Local reasoning model
            reasoning_config = LMStudioConfig(
                base_url=self.config.local_base_url,
                model_name=self.config.local_reasoning_model,
                temperature=0.7,
                max_tokens=2048
            )
            reasoning_client = LMStudioClient(reasoning_config)
            self.local_reasoning = ExpertLLMInterface(reasoning_client)
            logger.info(f"✓ Local reasoning fallback: {self.config.local_reasoning_model}")
            
            # Local action model
            action_config = LMStudioConfig(
                base_url=self.config.local_base_url,
                model_name=self.config.local_action_model,
                temperature=0.6,
                max_tokens=512
            )
            action_client = LMStudioClient(action_config)
            self.local_action = ExpertLLMInterface(action_client)
            logger.info(f"✓ Local action fallback: {self.config.local_action_model}")
            
        except Exception as e:
            logger.warning(f"Local fallback initialization failed: {e}")
            logger.info("Continuing without local fallbacks")
    
    def _report_initialization_status(self):
        """Report which models are available."""
        logger.info("=" * 70)
        logger.info("HYBRID LLM SYSTEM STATUS")
        logger.info("=" * 70)
        
        # Primary models
        logger.info("Primary Models:")
        logger.info(f"  Vision (Gemini): {'✓ Ready' if self.gemini else '✗ Unavailable'}")
        logger.info(f"  Reasoning (Claude): {'✓ Ready' if self.claude else '✗ Unavailable'}")
        
        # Fallback models
        if self.config.use_local_fallback:
            logger.info("Fallback Models:")
            logger.info(f"  Vision (Local): {'✓ Ready' if self.local_vision else '✗ Unavailable'}")
            logger.info(f"  Reasoning (Local): {'✓ Ready' if self.local_reasoning else '✗ Unavailable'}")
            logger.info(f"  Action (Local): {'✓ Ready' if self.local_action else '✗ Unavailable'}")
        else:
            logger.info("Fallback Models: Disabled")
        
        logger.info("=" * 70)
    
    async def analyze_image(
        self,
        prompt: str,
        image,
        temperature: float = 0.4,
        max_tokens: int = 768
    ) -> str:
        """
        Analyzes an image using the primary vision model (Gemini) with a fallback
        to a local vision model.

        Args:
            prompt (str): The prompt for the analysis.
            image: The image to analyze (e.g., a PIL Image object).
            temperature (float, optional): The sampling temperature. Defaults to 0.4.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 768.

        Raises:
            RuntimeError: If no vision model is available.

        Returns:
            str: The analysis of the image.
        """
        start_time = time.time()
        
        async with self.semaphore:
            # Rate limiting
            await self._rate_limit()
            
            # Try Gemini first with extended timeout and retries
            if self.gemini:
                try:
                    # Use longer timeout for Gemini vision (it's critical)
                    gemini_timeout = max(30, self.config.timeout)
                    result = await asyncio.wait_for(
                        self.gemini.analyze_image(
                            prompt=prompt,
                            image=image,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                            max_retries=3  # Enable retries
                        ),
                        timeout=gemini_timeout
                    )
                    
                    self.stats['gemini_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    # Log result with more detail
                    if result and len(result) > 0:
                        logger.debug(f"Gemini vision analysis: {len(result)} chars")
                        return result
                    else:
                        logger.warning("Gemini returned empty response (0 chars)")
                        # Fall through to local fallback
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Gemini vision timed out after {gemini_timeout}s")
                    self.stats['errors'] += 1
                    if not self.config.fallback_on_error:
                        raise
                    
                except Exception as e:
                    logger.warning(f"Gemini vision failed: {type(e).__name__}: {e}")
                    self.stats['errors'] += 1
                    
                    if not self.config.fallback_on_error:
                        raise
            
            # Try Gemini 2.5 Pro as fallback (superior quality, separate rate limit pool)
            if self.gemini and self.config.fallback_on_error:
                try:
                    logger.info("Trying Gemini 2.5 Pro vision fallback")
                    
                    # Create temporary Gemini 2.5 Pro client
                    import os as os_module
                    gemini_pro = GeminiClient(
                        api_key=os_module.getenv("GEMINI_API_KEY"),
                        model="gemini-2.5-pro",  # Gemini 2.5 Pro (best quality)
                        timeout=90  # Pro is slower, needs more time
                    )
                    
                    result = await asyncio.wait_for(
                        gemini_pro.analyze_image(
                            prompt=prompt,
                            image=image,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                            max_retries=2
                        ),
                        timeout=90
                    )
                    
                    await gemini_pro.close()
                    
                    if result and len(result) > 0:
                        logger.info(f"Gemini 2.5 Pro vision success: {len(result)} chars")
                        self.stats['gemini_calls'] += 1
                        self.stats['total_time'] += time.time() - start_time
                        return result
                    else:
                        logger.warning("Gemini 2.5 Pro returned empty response")
                        
                except Exception as e:
                    logger.warning(f"Gemini 2.5 Pro vision fallback failed: {type(e).__name__}: {e}")
                    # Continue to local fallback
            
            # Fallback to local vision model
            if self.local_vision and self.config.use_local_fallback:
                try:
                    logger.info("Using local vision fallback")
                    self.stats['fallback_activations'] += 1
                    
                    # Save image temporarily for local model
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        image.save(tmp.name, format='PNG')
                        tmp_path = tmp.name
                    
                    try:
                        response = await self.local_vision.generate(
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            image_path=tmp_path
                        )
                        
                        result = response.get('content', '')
                        self.stats['local_calls'] += 1
                        self.stats['total_time'] += time.time() - start_time
                        
                        return result
                    finally:
                        import os
                        os.unlink(tmp_path)
                        
                except Exception as e:
                    logger.error(f"Local vision fallback failed: {e}")
                    self.stats['errors'] += 1
                    raise
            
            # No vision model available
            raise RuntimeError("No vision model available (Gemini and local fallback both failed)")
    
    async def generate_reasoning(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generates reasoning using the primary reasoning model (Claude) with a
        fallback to a local reasoning model.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): An optional system prompt.
                                                          Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 2048.

        Raises:
            RuntimeError: If no reasoning model is available.

        Returns:
            str: The generated text.
        """
        start_time = time.time()
        
        async with self.semaphore:
            # Rate limiting
            await self._rate_limit()
            
            # Try Claude first
            if self.claude:
                try:
                    result = await asyncio.wait_for(
                        self.claude.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=self.config.timeout
                    )
                    
                    self.stats['claude_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    logger.debug(f"Claude reasoning: {len(result)} chars")
                    return result
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Claude reasoning timed out after {self.config.timeout}s")
                    self.stats['errors'] += 1
                    
                    if not self.config.fallback_on_error:
                        raise
                        
                except Exception as e:
                    error_msg = str(e) if str(e) else type(e).__name__
                    logger.warning(f"Claude reasoning failed: {error_msg}")
                    self.stats['errors'] += 1
                    
                    # Check if it's a rate limit error
                    if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                        logger.error("⚠️ Claude rate limit exceeded - consider increasing delays")
                    
                    if not self.config.fallback_on_error:
                        raise
            
            # Fallback to local reasoning model
            if self.local_reasoning and self.config.use_local_fallback:
                try:
                    logger.info("Using local reasoning fallback")
                    self.stats['fallback_activations'] += 1
                    
                    response = await self.local_reasoning.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    result = response.get('content', '')
                    self.stats['local_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Local reasoning fallback failed: {e}")
                    self.stats['errors'] += 1
                    raise
            
            # No reasoning model available
            raise RuntimeError("No reasoning model available (Claude and local fallback both failed)")
    
    async def generate_action(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 512
    ) -> str:
        """
        Generates a fast action decision using Claude or a local action model.

        Args:
            prompt (str): The prompt containing the action context.
            system_prompt (Optional[str], optional): An optional system prompt.
                                                          Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.6.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 512.

        Raises:
            RuntimeError: If no action model is available.

        Returns:
            str: The action decision.
        """
        start_time = time.time()
        
        async with self.semaphore:
            # Rate limiting
            await self._rate_limit()
            
            # Try Claude first (it's fast enough for actions)
            if self.claude:
                try:
                    result = await asyncio.wait_for(
                        self.claude.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=10.0  # Shorter timeout for actions
                    )
                    
                    self.stats['claude_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Claude action failed: {e}")
                    self.stats['errors'] += 1
                    
                    if not self.config.fallback_on_error:
                        raise
            
            # Fallback to local action model
            if self.local_action and self.config.use_local_fallback:
                try:
                    logger.info("Using local action fallback")
                    self.stats['fallback_activations'] += 1
                    
                    response = await self.local_action.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    result = response.get('content', '')
                    self.stats['local_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Local action fallback failed: {e}")
                    self.stats['errors'] += 1
                    raise
            
            # No action model available
            raise RuntimeError("No action model available")
    
    async def generate_world_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 4096
    ) -> str:
        """
        Generates a unified consciousness narrative using GPT-5-thinking, with a
        fallback to Claude.

        This method is intended to integrate all perspectives into a coherent,
        self-referential experience.

        Args:
            prompt (str): The integration prompt with all perspectives.
            system_prompt (Optional[str], optional): A system prompt defining the
                                                          consciousness role. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.8.
            max_tokens (int, optional): The maximum number of tokens to generate.
                                      Defaults to 4096.

        Raises:
            RuntimeError: If no world modeling model is available.

        Returns:
            str: The unified consciousness narrative.
        """
        start_time = time.time()
        
        async with self.semaphore:
            # Rate limiting
            await self._rate_limit()
            
            # Try GPT-5-thinking first
            if self.openai:
                try:
                    result = await asyncio.wait_for(
                        self.openai.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=self.config.timeout
                    )
                    
                    self.stats['openai_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    logger.debug(f"GPT-5-thinking world model: {len(result)} chars")
                    return result
                    
                except Exception as e:
                    logger.warning(f"GPT-5-thinking world model failed: {e}")
                    self.stats['errors'] += 1
                    
                    if not self.config.fallback_on_error:
                        raise
            
            # Fallback to Claude for world modeling
            if self.claude:
                try:
                    logger.info("Using Claude fallback for world modeling")
                    self.stats['fallback_activations'] += 1
                    
                    result = await asyncio.wait_for(
                        self.claude.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=self.config.timeout
                    )
                    
                    self.stats['claude_calls'] += 1
                    self.stats['total_time'] += time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Claude world model fallback failed: {e}")
                    self.stats['errors'] += 1
                    raise
            
            # No world modeling available
            raise RuntimeError("No world modeling available (GPT-5-thinking and Claude both unavailable)")
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.config.min_request_interval:
            await asyncio.sleep(self.config.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    async def close(self):
        """Closes all client connections."""
        if self.gemini:
            await self.gemini.close()
        if self.claude:
            await self.claude.close()
        if self.openai:
            await self.openai.close()
        
        logger.info("Hybrid LLM client closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of usage statistics for the client.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        total_calls = (
            self.stats['gemini_calls'] + 
            self.stats['claude_calls'] + 
            self.stats['openai_calls'] + 
            self.stats['local_calls']
        )
        
        return {
            **self.stats,
            'total_calls': total_calls,
            'avg_time': self.stats['total_time'] / max(1, total_calls),
            'primary_success_rate': (
                (self.stats['gemini_calls'] + self.stats['claude_calls'] + self.stats['openai_calls']) / 
                max(1, total_calls)
            ),
            'fallback_rate': self.stats['fallback_activations'] / max(1, total_calls)
        }
