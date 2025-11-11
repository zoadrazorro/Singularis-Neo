"""
Skyrim AGI Quickstart

Minimal example to get started with Skyrim AGI.

This is the simplest way to try it out!
"""

import asyncio
from singularis.skyrim import SkyrimAGI, SkyrimConfig


async def main():
    print("SINGULARIS AGI - SKYRIM QUICKSTART\n")

    # Create AGI with defaults
    config = SkyrimConfig(
        dry_run=True,  # Safe mode - won't control game
        autonomous_duration=60,  # 1 minute demo
    )

    agi = SkyrimAGI(config)

    # Optional: Initialize LLM for smarter decisions
    # await agi.initialize_llm()

    # Play!
    print("Starting autonomous gameplay for 1 minute...\n")
    await agi.autonomous_play()

    print("\n✓ Done! Check the output above to see what it learned.")


if __name__ == "__main__":
    asyncio.run(main())
