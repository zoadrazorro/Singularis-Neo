"""
Setup script for Singularis integrations.

Checks requirements, creates environment files, and validates setup.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check Python version is 3.10+."""
    print("Checking Python version...")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('aiohttp', 'aiohttp'),
        ('loguru', 'loguru'),
        ('pydantic', 'Pydantic'),
    ]
    
    missing = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} not installed")
            missing.append(package)
    
    if missing:
        print("\n⚠️ Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def check_singularis():
    """Check if Singularis is installed."""
    print("\nChecking Singularis installation...")
    
    try:
        import singularis
        print("✅ Singularis installed")
        return True
    except ImportError:
        print("❌ Singularis not installed")
        print("   Install with: cd .. && pip install -e .")
        return False


def create_env_file():
    """Create .env file template if it doesn't exist."""
    print("\nChecking .env file...")
    
    env_path = Path(".env")
    
    if env_path.exists():
        print("✅ .env file exists")
        return True
    
    print("Creating .env file template...")
    
    template = """# Singularis Integration Configuration
# Copy this file and fill in your actual values

# === Required API Keys ===

# OpenAI (for GPT-4/GPT-5)
OPENAI_API_KEY=your_openai_key_here

# Gemini (for vision processing)
GEMINI_API_KEY=your_gemini_key_here

# === Facebook Messenger ===

# Get these from: https://developers.facebook.com/
MESSENGER_PAGE_TOKEN=your_page_token_here
MESSENGER_VERIFY_TOKEN=your_custom_verify_token_here

# === Fitbit (Optional) ===

# Get these from: https://dev.fitbit.com/
# FITBIT_CLIENT_ID=your_client_id
# FITBIT_CLIENT_SECRET=your_client_secret

# === Meta Glasses (Optional) ===

# Enable Meta Glasses bridge
# ENABLE_META_GLASSES=false

# === Configuration ===

# Enable features
ENABLE_VISION=true
ENABLE_TTS=false
LEARNING_ENABLED=true

# Server settings
HOST=0.0.0.0
PORT=8080

# Database (future)
# DATABASE_URL=postgresql://user:password@localhost:5432/singularis
"""
    
    with open(env_path, 'w') as f:
        f.write(template)
    
    print("✅ Created .env template")
    print("⚠️ IMPORTANT: Edit .env and add your actual API keys!")
    return False  # Return False to indicate user needs to edit it


def check_api_keys():
    """Check if API keys are set in environment."""
    print("\nChecking API keys...")
    
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not installed, skipping .env load")
    
    keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'GEMINI_API_KEY': 'Gemini',
    }
    
    all_set = True
    
    for key, name in keys.items():
        value = os.getenv(key)
        if value and value != f'your_{key.lower()}':
            print(f"✅ {name} key set")
        else:
            print(f"❌ {name} key not set")
            all_set = False
    
    # Optional keys
    optional_keys = {
        'MESSENGER_PAGE_TOKEN': 'Messenger',
        'FITBIT_CLIENT_ID': 'Fitbit',
    }
    
    for key, name in optional_keys.items():
        value = os.getenv(key)
        if value and value != f'your_{key.lower()}':
            print(f"✅ {name} (optional) configured")
        else:
            print(f"⚠️ {name} (optional) not configured")
    
    return all_set


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        'logs',
        'data',
        'data/conversations',
        'data/health',
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"✅ Created {dir_name}/")
        else:
            print(f"✅ {dir_name}/ exists")
    
    return True


def print_next_steps():
    """Print next steps for user."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("""
1. Edit .env file and add your API keys:
   - OPENAI_API_KEY (required)
   - GEMINI_API_KEY (required)
   - MESSENGER_PAGE_TOKEN (for Messenger bot)

2. For Messenger bot, follow setup guide:
   - Read: MESSENGER_SETUP_GUIDE.md
   - Set up Facebook app and page
   - Configure webhook

3. For Fitbit integration:
   - Get credentials from dev.fitbit.com
   - Add to .env file
   - Complete OAuth flow

4. Test the setup:
   - Run: python test_messenger_bot.py
   - Start orchestrator: python main_orchestrator.py

5. Read documentation:
   - README.md - Complete integration guide
   - QUICK_START.md - Fast track setup
   - MESSENGER_SETUP_GUIDE.md - Messenger details

For help, see: integrations/README.md
""")


def main():
    """Run setup checks."""
    print("=" * 60)
    print("SINGULARIS INTEGRATIONS SETUP")
    print("=" * 60)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Singularis", check_singularis),
        ("Directories", create_directories),
        ("Environment file", create_env_file),
        ("API keys", check_api_keys),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 Setup complete! All checks passed.")
        print("   You're ready to start using the integrations.")
    else:
        print("\n⚠️ Some checks failed. Please address issues above.")
    
    print_next_steps()


if __name__ == "__main__":
    main()
