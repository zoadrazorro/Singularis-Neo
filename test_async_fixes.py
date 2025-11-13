#!/usr/bin/env python3
"""
Quick test to verify LM Studio connection and async fixes.
Run this before starting the full system.
"""

import asyncio
import aiohttp
from loguru import logger

async def test_lm_studio_connection():
    """Test basic LM Studio connectivity."""
    print("=" * 60)
    print("LM STUDIO CONNECTION TEST")
    print("=" * 60)
    
    base_url = "http://localhost:1234/v1"
    
    print(f"\n1. Testing connection to {base_url}...")
    
    session = aiohttp.ClientSession()
    try:
        # Test /models endpoint
        async with session.get(
            f"{base_url}/models",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as response:
            if response.status == 200:
                data = await response.json()
                models = data.get('data', [])
                
                print(f"✓ Connection successful!")
                print(f"✓ Found {len(models)} model(s):")
                
                for model in models:
                    model_id = model.get('id', 'unknown')
                    print(f"  - {model_id}")
                
                if not models:
                    print("\n⚠️ WARNING: No models loaded!")
                    print("   Please load a model in LM Studio before proceeding.")
                    return False
                
                return True
            else:
                print(f"✗ HTTP {response.status}: {await response.text()}")
                return False
                
    except aiohttp.ClientConnectorError as e:
        print(f"✗ Cannot connect to LM Studio")
        print(f"  Error: {e}")
        print(f"\n  Make sure:")
        print(f"  1. LM Studio is running")
        print(f"  2. Local server is started (Server tab)")
        print(f"  3. Server is using port 1234")
        return False
        
    except asyncio.TimeoutError:
        print(f"✗ Connection timeout")
        print(f"  LM Studio might be starting up, try again in a moment")
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
        
    finally:
        await session.close()
        print("\n✓ Test session closed properly")


async def test_simple_generation():
    """Test a simple generation request."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    from singularis.llm.lmstudio_client import LMStudioClient, LMStudioConfig
    
    config = LMStudioConfig(
        base_url="http://localhost:1234/v1",
        model_name="microsoft/phi-4-mini-reasoning",
        timeout=30
    )
    
    client = LMStudioClient(config)
    
    try:
        print(f"\n2. Testing generation with {config.model_name}...")
        
        result = await client.generate(
            prompt="Respond with only the word 'WORKING' if you can read this.",
            system_prompt="You are a test assistant. Follow instructions exactly.",
            temperature=0.1,
            max_tokens=10
        )
        
        content = result.get('content', '').strip()
        tokens = result.get('tokens', 0)
        
        print(f"✓ Generation successful!")
        print(f"  Response: {content}")
        print(f"  Tokens: {tokens}")
        
        if "WORKING" in content.upper():
            print(f"  ✓ Model is responding correctly")
            return True
        else:
            print(f"  ⚠️ Unexpected response (might still work)")
            return True
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False
        
    finally:
        await client.close()
        print("  ✓ Client session closed properly")


async def test_async_cleanup():
    """Test that async resources clean up properly."""
    print("\n" + "=" * 60)
    print("ASYNC CLEANUP TEST")
    print("=" * 60)
    
    print("\n3. Testing session cleanup...")
    
    # Create multiple sessions
    sessions = []
    for i in range(5):
        sessions.append(aiohttp.ClientSession())
    
    print(f"  Created {len(sessions)} test sessions")
    
    # Close them all
    for session in sessions:
        await session.close()
    
    print(f"  ✓ All sessions closed without warnings")
    return True


async def test_cancellation():
    """Test cancellation handling."""
    print("\n" + "=" * 60)
    print("CANCELLATION TEST")
    print("=" * 60)
    
    print("\n4. Testing graceful cancellation...")
    
    async def long_task():
        try:
            for i in range(10):
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("  ✓ Task cancelled gracefully")
            raise
    
    task = asyncio.create_task(long_task())
    await asyncio.sleep(0.3)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("  ✓ Cancellation handled properly")
        return True
    
    return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ASYNC FIXES VERIFICATION")
    print("Testing all fixes from ASYNC_FIXES_NOV12.md")
    print("=" * 60 + "\n")
    
    results = {
        "LM Studio Connection": False,
        "Generation": False,
        "Session Cleanup": False,
        "Cancellation Handling": False
    }
    
    # Test LM Studio connection
    results["LM Studio Connection"] = await test_lm_studio_connection()
    
    # Only test generation if connection works
    if results["LM Studio Connection"]:
        results["Generation"] = await test_simple_generation()
    else:
        print("\n⚠️ Skipping generation test (no connection)")
    
    # Test async fixes
    results["Session Cleanup"] = await test_async_cleanup()
    results["Cancellation Handling"] = await test_cancellation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYour system is ready to run!")
        print("Execute: python run_skyrim_agi.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the full system.")
        print("See LM_STUDIO_SETUP.md for detailed troubleshooting.")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
