"""
Service Launcher for Singularis Integrations

Starts all available services based on configuration.
Provides a simple way to launch everything at once.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_env():
    """Check if .env file exists and has keys."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Run: python setup.py")
        return False
    
    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not installed")
        print("   Install: pip install python-dotenv")
        return False
    
    # Check minimum required keys
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set in .env")
        return False
    
    return True


def start_main_orchestrator():
    """Start the main orchestrator (recommended)."""
    print("\n" + "=" * 60)
    print("STARTING MAIN ORCHESTRATOR")
    print("=" * 60)
    print("This starts all integrations in one service:")
    print("  - Messenger bot")
    print("  - Meta Glasses bridge")
    print("  - Fitbit health adapter")
    print("  - Unified API")
    print("\nPort: 8080")
    print("Access: http://localhost:8080")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run([sys.executable, "main_orchestrator.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Orchestrator stopped")


def start_messenger_only():
    """Start only Messenger bot."""
    print("\n" + "=" * 60)
    print("STARTING MESSENGER BOT ONLY")
    print("=" * 60)
    print("Port: 8000")
    print("Webhook: http://localhost:8000/webhook")
    print("\nUse ngrok for development:")
    print("  ngrok http 8000")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run([sys.executable, "messenger_bot_adapter.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Messenger bot stopped")


def start_glasses_only():
    """Start only Meta Glasses bridge."""
    print("\n" + "=" * 60)
    print("STARTING META GLASSES BRIDGE ONLY")
    print("=" * 60)
    print("Port: 8001")
    print("WebSocket: ws://localhost:8001/ws")
    print("HTTP API: http://localhost:8001/message")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run([sys.executable, "meta_glasses_bridge.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Glasses bridge stopped")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60 + "\n")
    
    subprocess.run([sys.executable, "test_messenger_bot.py"])
    
    print("\n✅ Tests complete")


def show_menu():
    """Show service launcher menu."""
    print("\n" + "=" * 60)
    print("SINGULARIS INTEGRATION LAUNCHER")
    print("=" * 60)
    print("\nWhat would you like to start?\n")
    print("1. Main Orchestrator (Recommended)")
    print("   All services in one - Messenger + Glasses + Fitbit + API")
    print("\n2. Messenger Bot Only")
    print("   Just the Facebook Messenger integration")
    print("\n3. Meta Glasses Bridge Only")
    print("   Just the glasses integration (requires extension)")
    print("\n4. Run Tests")
    print("   Test the Messenger bot locally")
    print("\n5. Exit")
    print("\n" + "=" * 60)
    
    choice = input("\nEnter choice (1-5): ").strip()
    return choice


def main():
    """Main launcher."""
    # Check environment first
    if not check_env():
        print("\n❌ Environment not configured properly")
        print("   Run: python setup.py")
        sys.exit(1)
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            start_main_orchestrator()
        elif choice == "2":
            start_messenger_only()
        elif choice == "3":
            start_glasses_only()
        elif choice == "4":
            run_tests()
            input("\nPress Enter to continue...")
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
