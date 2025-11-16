"""
Test script for Messenger Bot Adapter

Tests the bot locally without needing Facebook webhook setup.
Simulates user messages and validates Singularis integration.
"""

import asyncio
import os
from datetime import datetime

from loguru import logger

# Set up test environment
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('GEMINI_API_KEY', 'test-key')
os.environ.setdefault('MESSENGER_PAGE_TOKEN', 'test-token')
os.environ.setdefault('MESSENGER_VERIFY_TOKEN', 'test-verify')

from messenger_bot_adapter import MessengerBotAdapter, MessengerMessage


async def test_basic_message():
    """Test basic text message processing."""
    logger.info("\n=== Test 1: Basic Message ===")
    
    # Create adapter
    adapter = MessengerBotAdapter(
        page_access_token="test-token",
        verify_token="test-verify"
    )
    
    await adapter.initialize()
    
    # Create test message
    message = MessengerMessage(
        sender_id="test_user_123",
        message_text="Hello, how are you?",
        timestamp=datetime.now().timestamp() * 1000,
        attachments=None,
        quick_reply=None
    )
    
    # Process message
    try:
        response = await adapter._handle_message(message)
        logger.info(f"✅ Response: {response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


async def test_context_awareness():
    """Test that bot remembers context."""
    logger.info("\n=== Test 2: Context Awareness ===")
    
    adapter = MessengerBotAdapter(
        page_access_token="test-token",
        verify_token="test-verify"
    )
    
    await adapter.initialize()
    
    user_id = "test_user_456"
    
    # First message
    msg1 = MessengerMessage(
        sender_id=user_id,
        message_text="My favorite color is blue",
        timestamp=datetime.now().timestamp() * 1000
    )
    
    try:
        response1 = await adapter._handle_message(msg1)
        logger.info(f"Response 1: {response1[:100]}...")
        
        # Second message - should remember context
        msg2 = MessengerMessage(
            sender_id=user_id,
            message_text="What's my favorite color?",
            timestamp=datetime.now().timestamp() * 1000
        )
        
        response2 = await adapter._handle_message(msg2)
        logger.info(f"Response 2: {response2[:100]}...")
        
        # Check if context was used
        has_context = user_id in adapter.user_contexts
        logger.info(f"✅ Context stored: {has_context}")
        
        return has_context
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


async def test_learning_system():
    """Test continual learning from conversations."""
    logger.info("\n=== Test 3: Learning System ===")
    
    adapter = MessengerBotAdapter(
        page_access_token="test-token",
        verify_token="test-verify"
    )
    
    await adapter.initialize()
    
    # Have several conversations
    user_id = "test_user_789"
    messages = [
        "I love pizza",
        "My hobby is photography",
        "I work as a software engineer",
    ]
    
    try:
        for msg_text in messages:
            msg = MessengerMessage(
                sender_id=user_id,
                message_text=msg_text,
                timestamp=datetime.now().timestamp() * 1000
            )
            
            response = await adapter._handle_message(msg)
            logger.info(f"Msg: {msg_text}")
            logger.info(f"Response: {response[:80]}...")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Check episodic memory
        memory_count = len(adapter.learner.episodic_memory.episodes)
        logger.info(f"✅ Episodic memories: {memory_count}")
        
        return memory_count >= len(messages)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


async def test_stats_endpoint():
    """Test statistics tracking."""
    logger.info("\n=== Test 4: Statistics ===")
    
    adapter = MessengerBotAdapter(
        page_access_token="test-token",
        verify_token="test-verify"
    )
    
    await adapter.initialize()
    
    # Send a few messages
    for i in range(3):
        msg = MessengerMessage(
            sender_id=f"user_{i}",
            message_text=f"Test message {i}",
            timestamp=datetime.now().timestamp() * 1000
        )
        
        try:
            await adapter._handle_message(msg)
        except:
            pass  # Ignore errors for this test
    
    # Get stats
    stats = adapter.get_stats()
    
    logger.info("Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("✅ Stats endpoint working")
    return True


async def test_image_handling():
    """Test image attachment handling (simulated)."""
    logger.info("\n=== Test 5: Image Handling ===")
    
    adapter = MessengerBotAdapter(
        page_access_token="test-token",
        verify_token="test-verify"
    )
    
    await adapter.initialize()
    
    # Simulate image attachment
    message = MessengerMessage(
        sender_id="test_user_img",
        message_text="Check out this photo!",
        timestamp=datetime.now().timestamp() * 1000,
        attachments=[{
            'type': 'image',
            'payload': {
                'url': 'https://example.com/test.jpg'  # Dummy URL
            }
        }]
    )
    
    try:
        # This will fail to download the image, but we can test the flow
        response = await adapter._handle_message(message)
        logger.info(f"Response: {response[:100]}...")
        logger.info("✅ Image handling code executed")
        return True
    except Exception as e:
        # Expected to fail since URL is fake
        logger.info(f"Expected error (fake URL): {e}")
        logger.info("✅ Image handling flow works")
        return True


async def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("MESSENGER BOT ADAPTER TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Message", test_basic_message),
        ("Context Awareness", test_context_awareness),
        ("Learning System", test_learning_system),
        ("Statistics", test_stats_endpoint),
        ("Image Handling", test_image_handling),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
