import random
from pathlib import Path
from typing import Optional, Dict, Tuple

from singularis.skyrim.consciousness_bridge import ConsciousnessBridge
from singularis.llm.hybrid_client import HybridLLMClient


def _philosophy_texts_dir() -> Path:
    """Returns the path to the directory containing philosophy texts."""
    return Path(__file__).resolve().parents[2] / "philosophy_texts"


def _choose_random_text_file() -> Optional[Path]:
    """Selects a random text file from the philosophy texts directory.

    Returns:
        A Path object to a random file, or None if the directory or files
        do not exist.
    """
    pdir = _philosophy_texts_dir()
    if not pdir.exists():
        return None
    files = sorted([p for p in pdir.glob("*.txt") if p.is_file()])
    if not files:
        return None
    return random.choice(files)


def _extract_random_snippet(text: str, max_chars: int = 600) -> str:
    """Extracts a random paragraph or snippet from a larger body of text.

    Args:
        text: The source text.
        max_chars: The maximum character length for the snippet.

    Returns:
        A random snippet of the text.
    """
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        parts = [line.strip() for line in text.splitlines() if line.strip()]
    if not parts:
        return text[:max_chars]
    snippet = random.choice(parts)
    if len(snippet) <= max_chars:
        return snippet
    start = random.randint(0, max(0, len(snippet) - max_chars))
    end = start + max_chars
    return snippet[start:end]


async def get_random_philosophical_context(
    hybrid_llm: Optional[HybridLLMClient] = None,
    max_chars: int = 600,
) -> Dict[str, str]:
    """Injects a random piece of philosophical wisdom to guide the agent.

    This function selects a random text from the philosophy library, extracts a
    snippet, and optionally uses an LLM to summarize it into a concrete,
    actionable piece of guidance for the AGI.

    Args:
        hybrid_llm: An optional LLM client to summarize the philosophical excerpt.
        max_chars: The maximum length of the initial text snippet.

    Returns:
        A dictionary containing the source, the raw excerpt, and the final
        (potentially summarized) philosophical context.
    """
    path = _choose_random_text_file()
    if path is None:
        fallback = (
            "Ethics concerns our power to act coherently in the world. "
            "Seek actions that increase understanding, capability, and care."
        )
        return {
            "philosophical_source": "fallback",
            "philosophical_excerpt": fallback,
            "philosophical_context": fallback,
        }

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""

    excerpt = _extract_random_snippet(text or path.name, max_chars=max_chars)

    context_text = excerpt
    if hybrid_llm is not None:
        try:
            prompt = (
                "Summarize the following philosophy excerpt in 2-3 sentences focusing on ethical/"
                "worldview implications for moment-to-moment gameplay decisions. Keep it concrete.\n\n"
                f"EXCERPT:\n{excerpt}"
            )
            summary = await hybrid_llm.generate_reasoning(
                prompt=prompt,
                system_prompt=(
                    "You provide concise, actionable philosophical context to guide an agent's"
                    " awareness and decisions."
                ),
                temperature=0.3,
                max_tokens=220,
            )
            if isinstance(summary, str) and len(summary.strip()) > 0:
                context_text = summary.strip()
        except Exception:
            pass

    return {
        "philosophical_source": path.name,
        "philosophical_excerpt": excerpt,
        "philosophical_context": context_text,
    }


async def inform_consciousness_with_philosophy(
    bridge: ConsciousnessBridge,
    game_state: Dict,
    base_context: Optional[Dict] = None,
    hybrid_llm: Optional[HybridLLMClient] = None,
) -> Tuple:
    """Computes the AGI's consciousness state, augmented with philosophical context.

    This function orchestrates the process of fetching a random philosophical
    insight and integrating it into the context before computing the final
    consciousness state for the current cycle.

    Args:
        bridge: The ConsciousnessBridge to use for the computation.
        game_state: The current game state.
        base_context: The base context dictionary for the consciousness computation.
        hybrid_llm: An optional LLM client for summarizing the philosophy excerpt.

    Returns:
        A tuple containing the computed consciousness state and the full context
        used for the computation.
    """
    ctx = dict(base_context or {})
    phil = await get_random_philosophical_context(hybrid_llm=hybrid_llm)
    ctx.update(phil)
    state = await bridge.compute_consciousness(game_state, context=ctx)
    return state, ctx
