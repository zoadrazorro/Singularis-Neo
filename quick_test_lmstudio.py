"""
Quick test for the system prompt fix with LM Studio.

This script performs a simple, isolated test to verify that the `LMStudioClient`
can correctly send a request that includes both a user prompt and a system
prompt to a locally-hosted model via the LM Studio server.

It works by:
1.  Configuring and initializing an `LMStudioClient` to connect to a specific
    model (Mistral-7B Instruct v0.3 in this case).
2.  Calling the `generate` method with both a `prompt` and a `system_prompt`.
3.  Printing the result to confirm that the generation was successful and that
    the model respected the system prompt's instructions for brevity.
4.  Exiting with a status code of 0 on success or 1 on failure, making it
    suitable for use in automated testing or CI/CD pipelines.
"""

import asyncio
from singularis.llm.lmstudio_client import LMStudioClient, LMStudioConfig


async def quick_test():
    """Sets up the client and runs the test generation."""
    print("Testing Mistral-7B with system prompt...")
    config = LMStudioConfig(
        model_name="mistralai/mistral-7b-instruct-v0.3",
        max_tokens=100,
        timeout=10
    )
    client = LMStudioClient(config)
    
    try:
        result = await client.generate(
            prompt="Recommend ONE action from: jump, activate, move_forward",
            system_prompt="You are a Skyrim AI. Be very brief."
        )
        print(f"SUCCESS: {result['content']}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    exit(0 if success else 1)
