import random
from pathlib import Path
from typing import Optional, Dict, Tuple

from singularis.skyrim.consciousness_bridge import ConsciousnessBridge
from singularis.llm.hybrid_client import HybridLLMClient


def _philosophy_texts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "philosophy_texts"


def _choose_random_text_file() -> Optional[Path]:
    pdir = _philosophy_texts_dir()
    if not pdir.exists():
        return None
    files = sorted([p for p in pdir.glob("*.txt") if p.is_file()])
    if not files:
        return None
    return random.choice(files)


def _extract_random_snippet(text: str, max_chars: int = 600) -> str:
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
    ctx = dict(base_context or {})
    phil = await get_random_philosophical_context(hybrid_llm=hybrid_llm)
    ctx.update(phil)
    state = await bridge.compute_consciousness(game_state, context=ctx)
    return state, ctx
