"""
Setup and verification script for the Singularis Skyrim AGI.

This script automates the process of checking and installing all required
dependencies for running the AGI. It verifies the Python version, installs
core and Skyrim-specific packages using pip, and ensures that all necessary
Singularis AGI modules are importable.

Usage:
    python setup_skyrim.py

The script provides a step-by-step report of the verification process and will
attempt to install any missing packages automatically. It exits with a status
code of 0 on success and 1 on failure.
"""

import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """Checks if a Python package is installed by trying to find its spec.

    Args:
        package_name: The name of the package as it is known by pip.
        import_name: The name used to import the package. If None, it is
                     assumed to be the same as `package_name`.

    Returns:
        True if the package is installed, False otherwise.
    """
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_package(package_name):
    """Installs a Python package using pip.

    Args:
        package_name: The name of the package to install.
    """
    print(f"  Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def main():
    """Runs the main setup and verification process.

    This function executes a series of checks for the Python environment,
    dependencies, and internal AGI components, installing missing packages
    as needed and printing a final summary report.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 70)
    print("SINGULARIS SKYRIM AGI - SETUP & VERIFICATION")
    print("=" * 70)
    print()
    
    # Check Python version
    print("[1/5] Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"  [X] Python 3.10+ required, you have {sys.version}")
        return False
    print(f"  [OK] Python {sys.version_info.major}.{sys.version_info.minor}")
    print()
    
    # Check core dependencies
    print("[2/5] Checking core dependencies...")
    core_deps = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'networkx': 'networkx',
        'PIL': 'Pillow',
        'aiohttp': 'aiohttp',
        'loguru': 'loguru',
    }
    
    missing = []
    for import_name, package_name in core_deps.items():
        if check_package(package_name, import_name):
            print(f"  [OK] {package_name}")
        else:
            print(f"  [X] {package_name} (missing)")
            missing.append(package_name)
    
    if missing:
        print(f"\n  Installing {len(missing)} missing packages...")
        for pkg in missing:
            install_package(pkg)
    print()
    
    # Check Skyrim-specific dependencies
    print("[3/5] Checking Skyrim dependencies...")
    skyrim_deps = {
        'mss': 'mss',
        'pyautogui': 'pyautogui',
    }
    
    missing = []
    for import_name, package_name in skyrim_deps.items():
        if check_package(package_name, import_name):
            print(f"  [OK] {package_name}")
        else:
            print(f"  [X] {package_name} (missing)")
            missing.append(package_name)
    
    if missing:
        print(f"\n  Installing {len(missing)} missing packages...")
        for pkg in missing:
            install_package(pkg)
    print()
    
    # Check CLIP
    print("[4/5] Checking CLIP (vision model)...")
    if check_package('clip'):
        print("  [OK] CLIP installed")
    else:
        print("  [X] CLIP not installed")
        print("  Installing CLIP from GitHub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/openai/CLIP.git"
        ])
        print("  [OK] CLIP installed")
    print()
    
    # Verify Singularis components
    print("[5/5] Verifying Singularis components...")
    
    components = [
        ('singularis.skyrim', 'Skyrim integration'),
        ('singularis.agi_orchestrator', 'AGI orchestrator'),
        ('singularis.world_model', 'World model'),
        ('singularis.agency', 'Agency system'),
        ('singularis.learning', 'Continual learning'),
        ('singularis.neurosymbolic', 'Neurosymbolic reasoning'),
        ('singularis.active_inference', 'Active inference'),
        ('singularis.tier1_orchestrator', 'Consciousness engine'),
        ('singularis.llm', 'LLM integration'),
    ]
    
    all_ok = True
    for module_name, description in components:
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {description}")
        except ImportError as e:
            print(f"  [X] {description} - {e}")
            all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("[OK] ALL COMPONENTS VERIFIED!")
        print()
        print("Next steps:")
        print("  1. Start Skyrim")
        print("  2. Load a save in a safe location")
        print("  3. Run: python examples/skyrim_quickstart.py")
        print()
        print("For full guide, see: SKYRIM_SETUP.md")
    else:
        print("[X] SOME COMPONENTS MISSING")
        print()
        print("This likely means some AGI modules haven't been created yet.")
        print("Check which modules are missing above.")
    
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
