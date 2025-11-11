"""Tier-2 Specialized Expert Modules."""

# Template-based experts
from .base import Expert
from .reasoning_expert import ReasoningExpert
from .creative_expert import CreativeExpert
from .philosophical_expert import PhilosophicalExpert
from .technical_expert import TechnicalExpert
from .memory_expert import MemoryExpert
from .synthesis_expert import SynthesisExpert

# LLM-integrated experts
from .reasoning_expert_llm import ReasoningExpertLLM
from .creative_expert_llm import CreativeExpertLLM
from .philosophical_expert_llm import PhilosophicalExpertLLM
from .technical_expert_llm import TechnicalExpertLLM
from .memory_expert_llm import MemoryExpertLLM
from .synthesis_expert_llm import SynthesisExpertLLM

__all__ = [
    # Base
    "Expert",
    # Template-based
    "ReasoningExpert",
    "CreativeExpert",
    "PhilosophicalExpert",
    "TechnicalExpert",
    "MemoryExpert",
    "SynthesisExpert",
    # LLM-integrated
    "ReasoningExpertLLM",
    "CreativeExpertLLM",
    "PhilosophicalExpertLLM",
    "TechnicalExpertLLM",
    "MemoryExpertLLM",
    "SynthesisExpertLLM",
]
