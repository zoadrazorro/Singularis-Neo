"""LLM integration module for Singularis."""

from .lmstudio_client import (
    LMStudioClient,
    LMStudioConfig,
    ExpertLLMInterface,
)
from .claude_client import ClaudeClient
from .gemini_client import GeminiClient
from .hybrid_client import HybridLLMClient, HybridConfig, TaskType
from .moe_orchestrator import (
    MoEOrchestrator,
    ExpertRole,
    ExpertConfig,
    ExpertResponse,
    MoEResponse,
)
from .gpt_realtime_client import (
    GPTRealtimeClient,
    RealtimeConfig,
    RealtimeDecision,
    SubsystemType,
)
from .gpt5_orchestrator import (
    GPT5Orchestrator,
    SystemType,
    SystemMessage,
    GPT5Response,
)
from .hyperbolic_client import HyperbolicClient

__all__ = [
    "LMStudioClient",
    "LMStudioConfig",
    "ExpertLLMInterface",
    "ClaudeClient",
    "GeminiClient",
    "HybridLLMClient",
    "HybridConfig",
    "TaskType",
    "MoEOrchestrator",
    "ExpertRole",
    "ExpertConfig",
    "ExpertResponse",
    "MoEResponse",
    "GPTRealtimeClient",
    "RealtimeConfig",
    "RealtimeDecision",
    "SubsystemType",
    "GPT5Orchestrator",
    "SystemType",
    "SystemMessage",
    "GPT5Response",
    "HyperbolicClient",
]
