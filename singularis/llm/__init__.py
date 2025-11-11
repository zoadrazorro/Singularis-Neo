"""LLM integration module for Singularis."""

from .lmstudio_client import (
    LMStudioClient,
    LMStudioConfig,
    ExpertLLMInterface,
)

__all__ = [
    "LMStudioClient",
    "LMStudioConfig",
    "ExpertLLMInterface",
]
