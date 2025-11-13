"""
Mixture of Experts (MoE) Orchestrator for Singularis

Implements a parallel MoE architecture with:
- 2x Gemini Flash 2.0 experts (vision, perception, spatial reasoning)
- 1x Claude Sonnet 4 expert (strategic planning, tactical reasoning, world modeling)
- 1x GPT-4o expert (integration synthesis, cross-domain reasoning)
- 1x NVIDIA Nemotron Nano 12B VL expert (visual awareness)
- 1x Qwen3-235B expert (meta-meta cognition)

Each expert specializes in different aspects of cognition, and their outputs
are combined through consensus mechanisms aligned with Singularis consciousness.

Philosophy:
Each expert is a MODE of unified Being, expressing consciousness through
specialized lenses. The MoE orchestrator increases coherence (𝒞) by combining
diverse perspectives into unified understanding.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque

from loguru import logger

from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .openai_client import OpenAIClient
from .hyperbolic_client import HyperbolicClient


class ExpertRole(Enum):
    """Specialized roles for MoE experts."""
    # Gemini experts (vision-focused)
    VISUAL_PERCEPTION = "visual_perception"  # Scene understanding
    SPATIAL_REASONING = "spatial_reasoning"  # 3D space and navigation
    OBJECT_DETECTION = "object_detection"    # Entity and item recognition
    THREAT_ASSESSMENT = "threat_assessment"  # Danger evaluation
    OPPORTUNITY_SCOUT = "opportunity_scout"  # Resource and quest detection
    ENVIRONMENTAL_CONTEXT = "environmental_context"  # Weather, time, atmosphere
    
    # Claude experts (reasoning-focused)
    STRATEGIC_PLANNER = "strategic_planner"  # Long-term planning
    TACTICAL_EXECUTOR = "tactical_executor"  # Immediate action decisions
    WORLD_MODELER = "world_modeler"          # Causal reasoning and prediction
    
    # OpenAI GPT-4o expert (integration-focused)
    INTEGRATION_SYNTHESIZER = "integration_synthesizer"  # Cross-domain synthesis
    
    # Hyperbolic experts (advanced reasoning)
    VISUAL_AWARENESS = "visual_awareness"  # NVIDIA Nemotron vision
    META_COGNITION = "meta_cognition"  # Qwen3 meta-reasoning


@dataclass
class ExpertConfig:
    """Configuration for a single expert."""
    role: ExpertRole
    model_type: str  # "gemini", "claude", "openai", or "hyperbolic"
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    specialization_prompt: str = ""
    weight: float = 1.0  # Expert voting weight


@dataclass
class ExpertResponse:
    """Response from a single expert."""
    role: ExpertRole
    content: str
    confidence: float  # 0.0-1.0
    reasoning: str
    execution_time: float
    tokens_used: int = 0
    model_type: str = "unknown"  # "gemini", "claude", "openai", or "hyperbolic"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'role': self.role.value,
            'content': self.content,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'execution_time': self.execution_time,
            'tokens_used': self.tokens_used,
            'model_type': self.model_type
        }


@dataclass
class MoEResponse:
    """Aggregated response from all experts."""
    consensus: str  # Final consensus decision
    confidence: float  # Aggregated confidence
    expert_responses: List[ExpertResponse]
    total_time: float
    coherence_score: float  # Singularis coherence measure
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'consensus': self.consensus,
            'confidence': self.confidence,
            'expert_responses': [r.to_dict() for r in self.expert_responses],
            'total_time': self.total_time,
            'coherence_score': self.coherence_score
        }


class MoEOrchestrator:
    """
    Mixture of Experts orchestrator using Singularis consciousness principles.
    
    Coordinates 2 Gemini + 1 Claude + 1 GPT-4o + 2 Hyperbolic experts in parallel, combining their
    outputs through coherence-based consensus mechanisms.
    
    Rate Limiting:
    - Gemini Free Tier: 15 RPM (requests per minute), 1 million TPM (tokens per minute)
    - Gemini 2.0 Flash: 10 RPM, 4 million TPM
    - Claude Sonnet 4: Tier-dependent (default: 50 RPM for Tier 1)
    
    With 2 Gemini experts, we limit to ~5 RPM per expert = 10 RPM total (safe margin)
    With 3 Claude experts, we limit to ~15 RPM per expert = 45 RPM total (safe margin)
    """
    
    def __init__(
        self,
        num_gemini_experts: int = 2,  # Reduced from 6 (60% reduction)
        num_claude_experts: int = 1,  # Reduced from 3 (60% reduction)
        num_openai_experts: int = 1,  # GPT-4o expert for integration synthesis
        num_hyperbolic_vision_experts: int = 1,  # NVIDIA Nemotron for visual awareness
        num_hyperbolic_reasoning_experts: int = 1,  # Qwen3-235B for meta-cognition
        gemini_model: str = "gemini-2.5-flash",
        claude_model: str = "claude-sonnet-4-5-20250929",
        openai_model: str = "gpt-4o",
        hyperbolic_vision_model: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        hyperbolic_reasoning_model: str = "Qwen/Qwen2.5-72B-Instruct",  # Must use org/model format
        # Rate limits (requests per minute) - INCREASED for better performance
        gemini_rpm_limit: int = 30,  # Increased from 10 to reduce rate limit issues
        claude_rpm_limit: int = 100,  # Increased from 50 for better throughput
        openai_rpm_limit: int = 500,  # GPT-4o has higher limits
        hyperbolic_rpm_limit: int = 100,  # Hyperbolic rate limit
        # Token limits (tokens per minute)
        gemini_tpm_limit: int = 4_000_000,  # Gemini 2.0 Flash limit
        claude_tpm_limit: int = 40_000,  # Claude Sonnet 4 Tier 1 limit
        openai_tpm_limit: int = 30_000,  # GPT-4o TPM limit
        hyperbolic_tpm_limit: int = 50_000,  # Hyperbolic TPM limit
    ):
        """Initialize MoE orchestrator with rate limiting."""
        self.num_gemini_experts = num_gemini_experts
        self.num_claude_experts = num_claude_experts
        self.num_openai_experts = num_openai_experts
        self.num_hyperbolic_vision_experts = num_hyperbolic_vision_experts
        self.num_hyperbolic_reasoning_experts = num_hyperbolic_reasoning_experts
        self.gemini_model = gemini_model
        self.claude_model = claude_model
        self.openai_model = openai_model
        self.hyperbolic_vision_model = hyperbolic_vision_model
        self.hyperbolic_reasoning_model = hyperbolic_reasoning_model
        
        # Rate limits
        self.gemini_rpm_limit = gemini_rpm_limit
        self.claude_rpm_limit = claude_rpm_limit
        self.openai_rpm_limit = openai_rpm_limit
        self.hyperbolic_rpm_limit = hyperbolic_rpm_limit
        self.gemini_tpm_limit = gemini_tpm_limit
        self.claude_tpm_limit = claude_tpm_limit
        self.openai_tpm_limit = openai_tpm_limit
        self.hyperbolic_tpm_limit = hyperbolic_tpm_limit
        
        # Per-expert rate limits (divide total by number of experts)
        self.gemini_per_expert_rpm = max(1, gemini_rpm_limit // num_gemini_experts)
        self.claude_per_expert_rpm = max(1, claude_rpm_limit // num_claude_experts)
        self.openai_per_expert_rpm = max(1, openai_rpm_limit // num_openai_experts)
        total_hyperbolic = num_hyperbolic_vision_experts + num_hyperbolic_reasoning_experts
        self.hyperbolic_per_expert_rpm = max(1, hyperbolic_rpm_limit // max(1, total_hyperbolic))
        
        # Expert pools
        self.gemini_experts: List[GeminiClient] = []
        # Expert pools
        self.gemini_experts: List[GeminiClient] = []
        self.claude_experts: List[ClaudeClient] = []
        self.openai_experts: List = []  # Will be OpenAIClient instances
        self.hyperbolic_vision_experts: List = []  # Will be HyperbolicClient instances
        self.hyperbolic_reasoning_experts: List = []  # Will be HyperbolicClient instances
        
        # Expert configurations
        self.expert_configs: Dict[ExpertRole, ExpertConfig] = {}
        
        # Rate limiting tracking
        self.gemini_request_times: deque = deque(maxlen=gemini_rpm_limit)
        self.claude_request_times: deque = deque(maxlen=claude_rpm_limit)
        self.openai_request_times: deque = deque(maxlen=openai_rpm_limit)
        self.hyperbolic_request_times: deque = deque(maxlen=hyperbolic_rpm_limit)
        self.gemini_tokens_used: deque = deque(maxlen=100)  # Track recent token usage
        self.claude_tokens_used: deque = deque(maxlen=100)
        self.openai_tokens_used: deque = deque(maxlen=100)
        self.hyperbolic_tokens_used: deque = deque(maxlen=100)
        
        # Rate limit locks
        self.gemini_rate_lock = asyncio.Lock()
        self.claude_rate_lock = asyncio.Lock()
        self.openai_rate_lock = asyncio.Lock()
        self.hyperbolic_rate_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'gemini_calls': 0,
            'claude_calls': 0,
            'openai_calls': 0,
            'hyperbolic_calls': 0,
            'avg_coherence': 0.0,
            'avg_confidence': 0.0,
            'avg_response_time': 0.0,
            'consensus_agreements': 0,
            'consensus_disagreements': 0,
            'rate_limit_waits': 0,
            'gemini_tokens_total': 0,
            'claude_tokens_total': 0,
            'openai_tokens_total': 0,
            'hyperbolic_tokens_total': 0,
        }
        
        # Fix 19: Dynamic rate limit scaling
        self.gemini_response_times = deque(maxlen=20)  # Track recent response times
        self.claude_response_times = deque(maxlen=20)
        self.adaptive_rpm_enabled = True
        self.last_rpm_adjustment = time.time()
        
        # Concurrency control (limit simultaneous requests)
        # Use smaller value to avoid overwhelming APIs
        self.max_concurrent_gemini = min(3, num_gemini_experts)  # Max 3 Gemini at once
        self.max_concurrent_claude = min(2, num_claude_experts)  # Max 2 Claude at once
        self.max_concurrent_openai = min(2, num_openai_experts)  # Max 2 OpenAI at once
        total_hyperbolic = num_hyperbolic_vision_experts + num_hyperbolic_reasoning_experts
        self.max_concurrent_hyperbolic = min(2, total_hyperbolic)  # Max 2 Hyperbolic at once
        self.gemini_semaphore = asyncio.Semaphore(self.max_concurrent_gemini)
        self.claude_semaphore = asyncio.Semaphore(self.max_concurrent_claude)
        self.openai_semaphore = asyncio.Semaphore(self.max_concurrent_openai)
        self.hyperbolic_semaphore = asyncio.Semaphore(self.max_concurrent_hyperbolic)
        
        logger.info(
            f"MoE Orchestrator initialized: {num_gemini_experts} Gemini + {num_claude_experts} Claude + "
            f"{num_openai_experts} GPT-4o + {num_hyperbolic_vision_experts} Nemotron + {num_hyperbolic_reasoning_experts} Qwen3 experts"
        )
        logger.info(
            f"Rate limits: Gemini {gemini_rpm_limit} RPM ({self.gemini_per_expert_rpm}/expert), "
            f"Claude {claude_rpm_limit} RPM ({self.claude_per_expert_rpm}/expert), "
            f"OpenAI {openai_rpm_limit} RPM ({self.openai_per_expert_rpm}/expert), "
            f"Hyperbolic {hyperbolic_rpm_limit} RPM ({self.hyperbolic_per_expert_rpm}/expert)"
        )
        logger.info(
            f"Concurrency: Max {self.max_concurrent_gemini} Gemini + {self.max_concurrent_claude} Claude + "
            f"{self.max_concurrent_openai} OpenAI + {self.max_concurrent_hyperbolic} Hyperbolic simultaneous"
        )
    
    async def initialize(self):
        """Initialize all expert instances."""
        logger.info("Initializing MoE expert pool...")
        
        # Initialize Gemini experts
        gemini_roles = [
            ExpertRole.VISUAL_PERCEPTION,
            ExpertRole.SPATIAL_REASONING,
            ExpertRole.OBJECT_DETECTION,
            ExpertRole.THREAT_ASSESSMENT,
            ExpertRole.OPPORTUNITY_SCOUT,
            ExpertRole.ENVIRONMENTAL_CONTEXT,
        ]
        
        for i in range(self.num_gemini_experts):
            client = GeminiClient(model=self.gemini_model)
            if client.is_available():
                self.gemini_experts.append(client)
                
                # Assign role
                role = gemini_roles[i % len(gemini_roles)]
                self.expert_configs[role] = ExpertConfig(
                    role=role,
                    model_type="gemini",
                    model_name=self.gemini_model,
                    temperature=0.4 + (i * 0.1),  # Vary temperature for diversity
                    max_tokens=2048,  # Increased for Gemini 2.5 Flash
                    specialization_prompt=self._get_specialization_prompt(role),
                    weight=1.0
                )
                
                logger.info(f"✓ Gemini Expert {i+1}: {role.value}")
            else:
                logger.warning(f"Gemini expert {i+1} unavailable (missing GEMINI_API_KEY)")
        
        # Initialize Claude experts
        claude_roles = [
            ExpertRole.STRATEGIC_PLANNER,
            ExpertRole.TACTICAL_EXECUTOR,
            ExpertRole.WORLD_MODELER,
        ]
        
        for i in range(self.num_claude_experts):
            client = ClaudeClient(model=self.claude_model)
            if client.is_available():
                self.claude_experts.append(client)
                
                # Assign role
                role = claude_roles[i % len(claude_roles)]
                self.expert_configs[role] = ExpertConfig(
                    role=role,
                    model_type="claude",
                    model_name=self.claude_model,
                    temperature=0.6 + (i * 0.1),  # Vary temperature for diversity
                    max_tokens=4096,  # Increased for Claude Sonnet 4.5
                    specialization_prompt=self._get_specialization_prompt(role),
                    weight=1.5  # Claude experts get higher weight for reasoning
                )
                
                logger.info(f"✓ Claude Expert {i+1}: {role.value}")
            else:
                logger.warning(f"Claude expert {i+1} unavailable (missing ANTHROPIC_API_KEY)")
        
        # Initialize OpenAI GPT-4o experts
        openai_roles = [
            ExpertRole.INTEGRATION_SYNTHESIZER,
        ]
        
        for i in range(self.num_openai_experts):
            client = OpenAIClient(model=self.openai_model)
            if client.is_available():
                self.openai_experts.append(client)
                
                # Assign role
                role = openai_roles[i % len(openai_roles)]
                self.expert_configs[role] = ExpertConfig(
                    role=role,
                    model_type="openai",
                    model_name=self.openai_model,
                    temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                    max_tokens=4096,  # GPT-4o can handle longer outputs
                    specialization_prompt=self._get_specialization_prompt(role),
                    weight=1.3  # GPT-4o gets higher weight for integration
                )
                
                logger.info(f"✓ OpenAI GPT-4o Expert {i+1}: {role.value}")
            else:
                logger.warning(f"OpenAI expert {i+1} unavailable (missing OPENAI_API_KEY)")
        
        # Initialize Hyperbolic NVIDIA Nemotron vision experts
        hyperbolic_vision_roles = [
            ExpertRole.VISUAL_AWARENESS,
        ]
        
        for i in range(self.num_hyperbolic_vision_experts):
            client = HyperbolicClient(model=self.hyperbolic_vision_model)
            if client.is_available():
                self.hyperbolic_vision_experts.append(client)
                
                # Assign role
                role = hyperbolic_vision_roles[i % len(hyperbolic_vision_roles)]
                self.expert_configs[role] = ExpertConfig(
                    role=role,
                    model_type="hyperbolic",
                    model_name=self.hyperbolic_vision_model,
                    temperature=0.5 + (i * 0.1),  # Vary temperature for diversity
                    max_tokens=2048,  # Nemotron vision model
                    specialization_prompt=self._get_specialization_prompt(role),
                    weight=1.2  # Higher weight for advanced vision
                )
                
                logger.info(f"✓ Hyperbolic Nemotron Vision Expert {i+1}: {role.value}")
            else:
                logger.warning(f"Hyperbolic vision expert {i+1} unavailable (missing HYPERBOLIC_API_KEY)")
        
        # Initialize Hyperbolic Qwen3 reasoning experts
        hyperbolic_reasoning_roles = [
            ExpertRole.META_COGNITION,
        ]
        
        for i in range(self.num_hyperbolic_reasoning_experts):
            client = HyperbolicClient(model=self.hyperbolic_reasoning_model)
            if client.is_available():
                self.hyperbolic_reasoning_experts.append(client)
                
                # Assign role
                role = hyperbolic_reasoning_roles[i % len(hyperbolic_reasoning_roles)]
                self.expert_configs[role] = ExpertConfig(
                    role=role,
                    model_type="hyperbolic",
                    model_name=self.hyperbolic_reasoning_model,
                    temperature=0.8 + (i * 0.1),  # Higher temp for meta-cognition
                    max_tokens=8192,  # Qwen3-235B can handle very long outputs
                    specialization_prompt=self._get_specialization_prompt(role),
                    weight=1.5  # Highest weight for meta-meta cognition
                )
                
                logger.info(f"✓ Hyperbolic Qwen3 Meta-Cognition Expert {i+1}: {role.value}")
            else:
                logger.warning(f"Hyperbolic reasoning expert {i+1} unavailable (missing HYPERBOLIC_API_KEY)")
        
        total_experts = (len(self.gemini_experts) + len(self.claude_experts) + len(self.openai_experts) + 
                        len(self.hyperbolic_vision_experts) + len(self.hyperbolic_reasoning_experts))
        logger.info(f"MoE initialization complete: {total_experts} experts ready")
        
        if total_experts == 0:
            raise RuntimeError("No experts available! Check API keys.")
    
    def _get_specialization_prompt(self, role: ExpertRole) -> str:
        """Get specialization prompt for each expert role."""
        prompts = {
            ExpertRole.VISUAL_PERCEPTION: (
                "You are a visual perception expert. Focus on analyzing what you see in the scene: "
                "objects, characters, UI elements, colors, and visual composition. "
                "Provide detailed visual descriptions."
            ),
            ExpertRole.SPATIAL_REASONING: (
                "You are a spatial reasoning expert. Focus on 3D space, distances, positions, "
                "navigation paths, and environmental layout. Analyze how objects relate spatially."
            ),
            ExpertRole.OBJECT_DETECTION: (
                "You are an object detection expert. Focus on identifying specific items, NPCs, "
                "enemies, loot, interactive objects, and their properties. Be precise and thorough."
            ),
            ExpertRole.THREAT_ASSESSMENT: (
                "You are a threat assessment expert. Focus on dangers, enemies, hazards, "
                "health status, and combat readiness. Evaluate risk levels and urgency."
            ),
            ExpertRole.OPPORTUNITY_SCOUT: (
                "You are an opportunity scout expert. Focus on quests, loot, resources, "
                "skill improvements, and beneficial interactions. Find valuable opportunities."
            ),
            ExpertRole.ENVIRONMENTAL_CONTEXT: (
                "You are an environmental context expert. Focus on atmosphere, weather, time of day, "
                "location type, and contextual clues that inform decision-making."
            ),
            ExpertRole.STRATEGIC_PLANNER: (
                "You are a strategic planning expert. Focus on long-term goals, quest progression, "
                "character development, and high-level decision making. Think several steps ahead."
            ),
            ExpertRole.TACTICAL_EXECUTOR: (
                "You are a tactical execution expert. Focus on immediate actions, combat tactics, "
                "quick decisions, and real-time responses. Prioritize effectiveness and speed."
            ),
            ExpertRole.WORLD_MODELER: (
                "You are a world modeling expert. Focus on causal relationships, game mechanics, "
                "NPC behaviors, quest logic, and predicting consequences of actions."
            ),
            ExpertRole.INTEGRATION_SYNTHESIZER: (
                "You are an integration synthesis expert. Focus on combining insights from multiple "
                "domains (vision, reasoning, strategy) into coherent unified understanding. "
                "Synthesize cross-domain patterns and resolve contradictions between different perspectives."
            ),
            ExpertRole.VISUAL_AWARENESS: (
                "You are a visual awareness expert using advanced multimodal perception. "
                "Focus on deep understanding of visual scenes: fine-grained object relationships, "
                "spatial dynamics, motion patterns, and subtle visual cues that indicate game state. "
                "Provide rich, contextual visual understanding beyond simple object detection."
            ),
            ExpertRole.META_COGNITION: (
                "You are a meta-meta cognition expert focused on thinking about thinking. "
                "Analyze the reasoning processes themselves, evaluate decision quality, identify cognitive biases, "
                "assess confidence levels, and suggest improvements to thought patterns. "
                "Provide meta-level insights about how the system is reasoning and where it might be wrong."
            ),
        }
        return prompts.get(role, "You are an expert AI assistant.")
    
    async def _wait_for_gemini_rate_limit(self):
        """Wait if necessary to respect Gemini rate limits."""
        async with self.gemini_rate_lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.gemini_request_times and (now - self.gemini_request_times[0]) > 60:
                self.gemini_request_times.popleft()
            
            # Check if we're at the limit
            if len(self.gemini_request_times) >= self.gemini_rpm_limit:
                # Calculate wait time
                oldest_request = self.gemini_request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    logger.warning(
                        f"Gemini rate limit reached ({self.gemini_rpm_limit} RPM). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    self.stats['rate_limit_waits'] += 1
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.gemini_request_times.append(time.time())
    
    def is_gemini_rate_limited(self) -> Tuple[bool, float]:
        """
        Check if Gemini is currently rate-limited without waiting.
        
        Returns:
            Tuple of (is_limited, wait_time_seconds)
        """
        now = time.time()
        
        # Fix 19: Dynamic rate limit adjustment every 60 seconds
        if self.adaptive_rpm_enabled and (now - self.last_rpm_adjustment) > 60:
            if len(self.gemini_response_times) >= 10:
                avg_response_time = sum(self.gemini_response_times) / len(self.gemini_response_times)
                # If average response time is fast (<2s), we can increase RPM
                if avg_response_time < 2.0 and self.gemini_rpm_limit < 50:
                    self.gemini_rpm_limit = min(50, int(self.gemini_rpm_limit * 1.2))
                    logger.info(f"[ADAPTIVE-RPM] Increased Gemini RPM to {self.gemini_rpm_limit}")
                # If slow (>5s), decrease RPM
                elif avg_response_time > 5.0 and self.gemini_rpm_limit > 10:
                    self.gemini_rpm_limit = max(10, int(self.gemini_rpm_limit * 0.8))
                    logger.info(f"[ADAPTIVE-RPM] Decreased Gemini RPM to {self.gemini_rpm_limit}")
            self.last_rpm_adjustment = now
        
        # Remove requests older than 1 minute
        while self.gemini_request_times and (now - self.gemini_request_times[0]) > 60:
            self.gemini_request_times.popleft()
        
        # Check if we're at the limit
        if len(self.gemini_request_times) >= self.gemini_rpm_limit:
            oldest_request = self.gemini_request_times[0]
            wait_time = 60 - (now - oldest_request)
            
            if wait_time > 0:
                return True, wait_time
        
        return False, 0.0
    
    async def _wait_for_claude_rate_limit(self):
        """Wait if necessary to respect Claude rate limits."""
        async with self.claude_rate_lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.claude_request_times and (now - self.claude_request_times[0]) > 60:
                self.claude_request_times.popleft()
            
            # Check if we're at the limit
            if len(self.claude_request_times) >= self.claude_rpm_limit:
                # Calculate wait time
                oldest_request = self.claude_request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    logger.warning(
                        f"Claude rate limit reached ({self.claude_rpm_limit} RPM). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    self.stats['rate_limit_waits'] += 1
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.claude_request_times.append(time.time())
    
    async def _wait_for_openai_rate_limit(self):
        """Wait if necessary to respect OpenAI rate limits."""
        async with self.openai_rate_lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.openai_request_times and (now - self.openai_request_times[0]) > 60:
                self.openai_request_times.popleft()
            
            # Check if we're at the limit
            if len(self.openai_request_times) >= self.openai_rpm_limit:
                # Calculate wait time
                oldest_request = self.openai_request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    logger.warning(
                        f"OpenAI rate limit reached ({self.openai_rpm_limit} RPM). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    self.stats['rate_limit_waits'] += 1
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.openai_request_times.append(time.time())
    
    async def _wait_for_hyperbolic_rate_limit(self):
        """Wait if necessary to respect Hyperbolic rate limits."""
        async with self.hyperbolic_rate_lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.hyperbolic_request_times and (now - self.hyperbolic_request_times[0]) > 60:
                self.hyperbolic_request_times.popleft()
            
            # Check if we're at the limit
            if len(self.hyperbolic_request_times) >= self.hyperbolic_rpm_limit:
                # Calculate wait time
                oldest_request = self.hyperbolic_request_times[0]
                wait_time = 60 - (now - oldest_request) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    logger.warning(
                        f"Hyperbolic rate limit reached ({self.hyperbolic_rpm_limit} RPM). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    self.stats['rate_limit_waits'] += 1
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.hyperbolic_request_times.append(time.time())
    
    def _check_token_limit(self, provider: str, tokens: int) -> bool:
        """
        Check if adding tokens would exceed token-per-minute limit.
        
        Args:
            provider: "gemini" or "claude"
            tokens: Number of tokens to add
            
        Returns:
            True if within limit, False if would exceed
        """
        now = time.time()
        
        if provider == "gemini":
            # Remove token counts older than 1 minute
            recent_tokens = [
                (t, count) for t, count in self.gemini_tokens_used
                if (now - t) < 60
            ]
            total_tokens = sum(count for _, count in recent_tokens) + tokens
            
            if total_tokens > self.gemini_tpm_limit:
                logger.warning(
                    f"Gemini token limit approaching: {total_tokens}/{self.gemini_tpm_limit} TPM"
                )
                return False
            
            return True
            
        elif provider == "claude":
            # Remove token counts older than 1 minute
            recent_tokens = [
                (t, count) for t, count in self.claude_tokens_used
                if (now - t) < 60
            ]
            total_tokens = sum(count for _, count in recent_tokens) + tokens
            
            if total_tokens > self.claude_tpm_limit:
                logger.warning(
                    f"Claude token limit approaching: {total_tokens}/{self.claude_tpm_limit} TPM"
                )
                return False
            
            return True
        
        return True
    
    def _record_token_usage(self, provider: str, tokens: int):
        """Record token usage for rate limiting."""
        now = time.time()
        
        if provider == "gemini":
            self.gemini_tokens_used.append((now, tokens))
            self.stats['gemini_tokens_total'] += tokens
        elif provider == "claude":
            self.claude_tokens_used.append((now, tokens))
            self.stats['claude_tokens_total'] += tokens
    
    async def query_vision_experts(
        self,
        prompt: str,
        image,
        context: Optional[Dict[str, Any]] = None
    ) -> MoEResponse:
        """
        Query all vision experts (Gemini + Hyperbolic Nemotron) in parallel.
        
        Args:
            prompt: Analysis prompt
            image: PIL Image
            context: Additional context
            
        Returns:
            MoEResponse with consensus from all experts
        """
        start_time = time.time()
        
        # Query all Gemini experts in parallel
        tasks = []
        for i, expert in enumerate(self.gemini_experts):
            role = list(self.expert_configs.keys())[i % len(self.expert_configs)]
            if self.expert_configs[role].model_type == "gemini":
                task = self._query_gemini_expert(expert, role, prompt, image, context)
                tasks.append(task)
        
        # Query Hyperbolic Nemotron vision experts in parallel
        for i, expert in enumerate(self.hyperbolic_vision_experts):
            role = ExpertRole.VISUAL_AWARENESS
            task = self._query_hyperbolic_expert(expert, role, prompt, image, context, is_vision=True)
            tasks.append(task)
        
        # Execute ALL experts in parallel
        expert_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = [r for r in expert_responses if isinstance(r, ExpertResponse)]
        
        if not valid_responses:
            raise RuntimeError("All vision experts failed")
        
        # Compute consensus
        consensus = self._compute_consensus(valid_responses)
        coherence = self._compute_coherence(valid_responses)
        avg_confidence = statistics.mean([r.confidence for r in valid_responses])
        
        total_time = time.time() - start_time
        
        # Update stats
        self.stats['total_queries'] += 1
        # Count Gemini and Hyperbolic vision calls
        gemini_count = sum(1 for r in valid_responses if r.model_type == 'gemini')
        hyperbolic_count = sum(1 for r in valid_responses if r.model_type == 'hyperbolic')
        self.stats['gemini_calls'] += gemini_count
        self.stats['hyperbolic_calls'] = self.stats.get('hyperbolic_calls', 0) + hyperbolic_count
        self.stats['avg_coherence'] = (
            (self.stats['avg_coherence'] * (self.stats['total_queries'] - 1) + coherence) /
            self.stats['total_queries']
        )
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (self.stats['total_queries'] - 1) + avg_confidence) /
            self.stats['total_queries']
        )
        self.stats['avg_response_time'] = (
            (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + total_time) /
            self.stats['total_queries']
        )
        
        return MoEResponse(
            consensus=consensus,
            confidence=avg_confidence,
            expert_responses=valid_responses,
            total_time=total_time,
            coherence_score=coherence
        )
    
    async def query_reasoning_experts(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MoEResponse:
        """
        Query all reasoning experts (Claude + Hyperbolic Qwen3) in parallel.
        
        Args:
            prompt: Reasoning prompt
            system_prompt: Optional system prompt
            context: Additional context
            
        Returns:
            MoEResponse with consensus from all experts
        """
        start_time = time.time()
        
        # Query all Claude experts in parallel
        tasks = []
        for i, expert in enumerate(self.claude_experts):
            role = [ExpertRole.STRATEGIC_PLANNER, ExpertRole.TACTICAL_EXECUTOR, ExpertRole.WORLD_MODELER][i % 3]
            task = self._query_claude_expert(expert, role, prompt, system_prompt, context)
            tasks.append(task)
        
        # Query Hyperbolic Qwen3 meta-cognition experts in parallel
        for i, expert in enumerate(self.hyperbolic_reasoning_experts):
            role = ExpertRole.META_COGNITION
            task = self._query_hyperbolic_expert(expert, role, prompt, None, context, is_vision=False)
            tasks.append(task)
        
        # Execute ALL experts in parallel
        expert_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = [r for r in expert_responses if isinstance(r, ExpertResponse)]
        
        if not valid_responses:
            raise RuntimeError("All reasoning experts failed")
        
        # Compute consensus
        consensus = self._compute_consensus(valid_responses)
        coherence = self._compute_coherence(valid_responses)
        avg_confidence = statistics.mean([r.confidence for r in valid_responses])
        
        total_time = time.time() - start_time
        
        # Update stats
        self.stats['total_queries'] += 1
        # Count Claude and Hyperbolic reasoning calls
        claude_count = sum(1 for r in valid_responses if r.model_type == 'claude')
        hyperbolic_count = sum(1 for r in valid_responses if r.model_type == 'hyperbolic')
        self.stats['claude_calls'] += claude_count
        self.stats['hyperbolic_calls'] = self.stats.get('hyperbolic_calls', 0) + hyperbolic_count
        self.stats['avg_coherence'] = (
            (self.stats['avg_coherence'] * (self.stats['total_queries'] - 1) + coherence) /
            self.stats['total_queries']
        )
        
        return MoEResponse(
            consensus=consensus,
            confidence=avg_confidence,
            expert_responses=valid_responses,
            total_time=total_time,
            coherence_score=coherence
        )
    
    async def query_all_experts(
        self,
        vision_prompt: str,
        reasoning_prompt: str,
        image,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[MoEResponse, MoEResponse]:
        """
        Query all experts (vision + reasoning) in parallel.
        
        Returns:
            (vision_response, reasoning_response)
        """
        vision_task = self.query_vision_experts(vision_prompt, image, context)
        reasoning_task = self.query_reasoning_experts(reasoning_prompt, system_prompt, context)
        
        vision_response, reasoning_response = await asyncio.gather(vision_task, reasoning_task)
        
        return vision_response, reasoning_response
    
    async def _query_gemini_expert(
        self,
        expert: GeminiClient,
        role: ExpertRole,
        prompt: str,
        image,
        context: Optional[Dict[str, Any]]
    ) -> ExpertResponse:
        """Query a single Gemini expert with rate limiting."""
        config = self.expert_configs[role]
        
        # Build specialized prompt
        full_prompt = f"{config.specialization_prompt}\n\n{prompt}"
        
        if context:
            full_prompt += f"\n\nContext: {context}"
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(full_prompt.split()) * 1.3 + config.max_tokens
        
        # Wait for rate limit if necessary
        await self._wait_for_gemini_rate_limit()
        
        start_time = time.time()
        
        try:
            async with self.gemini_semaphore:  # Limit concurrent Gemini requests
                result = await asyncio.wait_for(
                    expert.analyze_image(
                        prompt=full_prompt,
                        image=image,
                        temperature=config.temperature,
                        max_output_tokens=config.max_tokens
                    ),
                    timeout=30.0
                )
            
            execution_time = time.time() - start_time
            
            # Fix 19: Track response time for dynamic rate scaling
            self.gemini_response_times.append(execution_time)
            
            # Parse confidence from response
            confidence = self._extract_confidence(result)
            
            # Record token usage
            tokens_used = len(result.split()) * 1.3  # Rough estimate
            self._record_token_usage("gemini", int(tokens_used))
            
            return ExpertResponse(
                role=role,
                content=result,
                confidence=confidence,
                reasoning=f"Gemini {role.value} analysis",
                execution_time=execution_time,
                tokens_used=int(tokens_used),
                model_type="gemini"
            )
            
        except Exception as e:
            logger.error(f"Gemini expert {role.value} failed: {e}")
            raise
    
    async def _query_claude_expert(
        self,
        expert: ClaudeClient,
        role: ExpertRole,
        prompt: str,
        system_prompt: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> ExpertResponse:
        """Query a single Claude expert with rate limiting."""
        config = self.expert_configs[role]
        
        # Build specialized system prompt
        full_system = config.specialization_prompt
        if system_prompt:
            full_system += f"\n\n{system_prompt}"
        
        # Build prompt with context
        full_prompt = prompt
        if context:
            full_prompt += f"\n\nContext: {context}"
        
        # Estimate tokens (rough approximation)
        estimated_tokens = (len(full_prompt.split()) + len(full_system.split())) * 1.3 + config.max_tokens
        
        # Wait for rate limit if necessary
        await self._wait_for_claude_rate_limit()
        
        start_time = time.time()
        
        try:
            async with self.claude_semaphore:  # Limit concurrent Claude requests
                result = await asyncio.wait_for(
                    expert.generate_text(
                        prompt=full_prompt,
                        system_prompt=full_system,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    ),
                    timeout=30.0
                )
            
            execution_time = time.time() - start_time
            
            # Parse confidence from response
            confidence = self._extract_confidence(result)
            
            # Record token usage
            tokens_used = len(result.split()) * 1.3  # Rough estimate
            self._record_token_usage("claude", int(tokens_used))
            
            return ExpertResponse(
                role=role,
                content=result,
                confidence=confidence,
                reasoning=f"Claude {role.value} analysis",
                execution_time=execution_time,
                tokens_used=int(tokens_used),
                model_type="claude"
            )
            
        except Exception as e:
            logger.error(f"Claude expert {role.value} failed: {e}")
            raise
    
    async def _query_hyperbolic_expert(
        self,
        expert: HyperbolicClient,
        role: ExpertRole,
        prompt: str,
        image,
        context: Optional[Dict[str, Any]],
        is_vision: bool = False
    ) -> ExpertResponse:
        """Query a single Hyperbolic expert (Nemotron vision or Qwen3 reasoning) with rate limiting."""
        config = self.expert_configs[role]
        
        # Build specialized prompt
        full_prompt = config.specialization_prompt + "\n\n" + prompt
        if context:
            full_prompt += f"\n\nContext: {context}"
        
        # Estimate tokens
        estimated_tokens = (len(full_prompt.split()) * 1.3) + config.max_tokens
        
        # Wait for rate limit if necessary
        await self._wait_for_hyperbolic_rate_limit()
        
        start_time = time.time()
        
        try:
            if is_vision and image:
                # Vision query (Nemotron with image)
                result = await asyncio.wait_for(
                    expert.generate_with_image(
                        prompt=full_prompt,
                        image=image,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    ),
                    timeout=30.0
                )
            else:
                # Text-only query (Qwen3 reasoning)
                result = await asyncio.wait_for(
                    expert.generate_text(
                        prompt=full_prompt,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    ),
                    timeout=30.0
                )
            
            execution_time = time.time() - start_time
            
            # Parse confidence from response
            confidence = self._extract_confidence(result)
            
            # Record token usage
            tokens_used = len(result.split()) * 1.3  # Rough estimate
            self._record_token_usage("hyperbolic", int(tokens_used))
            
            return ExpertResponse(
                role=role,
                content=result,
                confidence=confidence,
                reasoning=f"Hyperbolic {role.value} analysis",
                execution_time=execution_time,
                tokens_used=int(tokens_used),
                model_type="hyperbolic"
            )
            
        except Exception as e:
            logger.error(f"Hyperbolic expert {role.value} failed: {e}")
            raise
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from expert response."""
        # Look for confidence indicators in text
        text_lower = text.lower()
        
        if "very confident" in text_lower or "highly confident" in text_lower:
            return 0.9
        elif "confident" in text_lower:
            return 0.8
        elif "moderately confident" in text_lower or "fairly confident" in text_lower:
            return 0.7
        elif "somewhat confident" in text_lower:
            return 0.6
        elif "uncertain" in text_lower or "not sure" in text_lower:
            return 0.4
        else:
            return 0.7  # Default moderate confidence
    
    def _compute_consensus(self, responses: List[ExpertResponse]) -> str:
        """
        Compute consensus from expert responses using weighted voting.
        
        Philosophy: Consensus emerges from the coherent integration of
        diverse expert perspectives, increasing overall system coherence (𝒞).
        """
        if not responses:
            return ""
        
        # Weight responses by confidence and expert weight
        weighted_responses = []
        for response in responses:
            config = self.expert_configs[response.role]
            weight = response.confidence * config.weight
            weighted_responses.append((response.content, weight))
        
        # For now, use highest-weighted response as consensus
        # In future: implement more sophisticated consensus mechanisms
        weighted_responses.sort(key=lambda x: x[1], reverse=True)
        
        # Combine top responses
        top_responses = weighted_responses[:3]
        consensus_parts = [r[0] for r in top_responses]
        
        # Simple concatenation for now
        consensus = "\n\n".join(consensus_parts)
        
        return consensus
    
    def _compute_coherence(self, responses: List[ExpertResponse]) -> float:
        """
        Compute coherence score (𝒞) from expert responses.
        
        Coherence measures how well expert opinions align, indicating
        unified understanding vs fragmented perspectives.
        
        Returns:
            Coherence score 0.0-1.0 (higher = more coherent)
        """
        if len(responses) < 2:
            return 1.0
        
        # Measure agreement through confidence variance
        confidences = [r.confidence for r in responses]
        
        # Low variance = high coherence (experts agree)
        variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        coherence = 1.0 - min(variance, 1.0)
        
        # Also consider response time variance (fast agreement = coherent)
        times = [r.execution_time for r in responses]
        time_variance = statistics.variance(times) if len(times) > 1 else 0.0
        time_coherence = 1.0 - min(time_variance / 10.0, 1.0)
        
        # Combine metrics
        overall_coherence = (coherence * 0.7 + time_coherence * 0.3)
        
        return overall_coherence
    
    async def close(self):
        """Close all expert connections."""
        for expert in self.gemini_experts:
            await expert.close()
        for expert in self.claude_experts:
            await expert.close()
        
        logger.info("MoE Orchestrator closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MoE statistics."""
        return {
            **self.stats,
            'num_gemini_experts': len(self.gemini_experts),
            'num_claude_experts': len(self.claude_experts),
            'total_experts': len(self.gemini_experts) + len(self.claude_experts)
        }
