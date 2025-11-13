"""Quick test of voice system"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_voice():
    from singularis.consciousness.voice_system import VoiceSystem, VoiceType, ThoughtPriority
    
    print("Testing voice system...")
    print(f"GEMINI_API_KEY present: {bool(os.getenv('GEMINI_API_KEY'))}")
    
    voice = VoiceSystem(
        voice=VoiceType.CHARON,
        enabled=True,
        min_priority=ThoughtPriority.MEDIUM
    )
    
    print(f"Voice system enabled: {voice.enabled}")
    print(f"Voice: {voice.voice.value}")
    print(f"Min priority: {voice.min_priority.value}")
    
    print("\nSpeaking test message...")
    result = await voice.speak_decision("explore", "Testing voice system")
    print(f"Speech result: {result}")
    
    # Wait longer for audio to play (audio is ~10 seconds)
    print("Waiting for audio playback...")
    await asyncio.sleep(15)
    
    stats = voice.get_stats()
    print(f"\nStats: {stats}")
    
    await voice.close()

if __name__ == "__main__":
    asyncio.run(test_voice())
