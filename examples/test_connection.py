"""
Simple connection test for LM Studio.

Run this first to verify your LM Studio setup is working.

Usage:
    python examples/test_connection.py
"""

import asyncio
import aiohttp
from loguru import logger


async def test_lm_studio():
    """Test connection to LM Studio."""
    
    base_url = "http://localhost:1234/v1"
    
    logger.info("=" * 60)
    logger.info("LM Studio Connection Test")
    logger.info("=" * 60)
    
    # Test 1: Check if server is running
    logger.info("\n[Test 1] Checking if LM Studio server is running...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.success("✓ LM Studio server is running!")
                    
                    if data.get('data'):
                        model = data['data'][0]
                        logger.info(f"  Model loaded: {model.get('id', 'unknown')}")
                    else:
                        logger.warning("  No models loaded in LM Studio")
                else:
                    logger.error(f"✗ Server returned status {response.status}")
                    return False
    except aiohttp.ClientError as e:
        logger.error(f"✗ Cannot connect to LM Studio: {e}")
        logger.info("\nMake sure:")
        logger.info("  1. LM Studio is running")
        logger.info("  2. A model is loaded")
        logger.info("  3. Local server is started on port 1234")
        return False
    except asyncio.TimeoutError:
        logger.error("✗ Connection timeout")
        return False
    
    # Test 2: Send a simple completion request
    logger.info("\n[Test 2] Sending test completion request...")
    try:
        payload = {
            "model": "local-model",  # LM Studio accepts any model name
            "messages": [
                {"role": "user", "content": "Say 'Hello from Singularis!' and nothing else."}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    tokens = data.get('usage', {}).get('total_tokens', 0)
                    
                    logger.success("✓ Completion request successful!")
                    logger.info(f"  Response: {content}")
                    logger.info(f"  Tokens used: {tokens}")
                else:
                    logger.error(f"✗ Request failed with status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"✗ Completion request failed: {e}")
        return False
    
    # Test 3: Check response quality
    logger.info("\n[Test 3] Testing philosophical response...")
    try:
        payload = {
            "model": "local-model",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a philosophical assistant grounded in Spinoza's Ethics."
                },
                {
                    "role": "user",
                    "content": "In one sentence, what is Spinoza's concept of Substance?"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    
                    logger.success("✓ Philosophical response received!")
                    logger.info(f"  Response: {content}")
                    
                    # Check if response is reasonable
                    if len(content) > 20 and any(word in content.lower() for word in ['substance', 'infinite', 'god', 'nature', 'being']):
                        logger.success("✓ Response quality looks good!")
                    else:
                        logger.warning("⚠ Response may need better prompting")
                else:
                    logger.error(f"✗ Request failed with status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"✗ Philosophical test failed: {e}")
        return False
    
    # All tests passed
    logger.info("\n" + "=" * 60)
    logger.success("All tests passed! ✓")
    logger.info("=" * 60)
    logger.info("\nYour LM Studio setup is ready for Singularis!")
    logger.info("\nNext steps:")
    logger.info("  1. Run: python examples/quickstart_llm.py")
    logger.info("  2. Read: docs/LM_STUDIO_INTEGRATION.md")
    logger.info("  3. Create remaining 5 LLM experts")
    
    return True


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO"
    )
    
    # Run test
    success = asyncio.run(test_lm_studio())
    
    if not success:
        exit(1)
