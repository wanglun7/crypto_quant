#!/usr/bin/env python3
"""Check if demo pipeline requirements are met."""

import sys
from pathlib import Path


def check_requirements():
    """Check if all required dependencies are available."""
    print("Checking demo pipeline requirements...")

    missing_deps = []

    # Check Python version
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 11):
        print(f"⚠️  Python 3.11+ recommended (found {current_version})")
        print("   The demo may work but is not officially supported")
    else:
        print(f"✅ Python {current_version} OK")

    # Check required packages
    required_packages = [
        "structlog",
        "websockets",
        "asyncpg",
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing_deps.append(package)

    # Check if crypto_quant module can be imported
    try:
        from crypto_quant.data_pipeline.collectors.binance_ws import BinanceWSCollector
        from crypto_quant.data_pipeline.storage.writer import TimescaleWriter
        print("✅ crypto_quant modules importable")
    except ImportError as e:
        print(f"❌ crypto_quant modules not importable: {e}")
        return False

    # Check if demo script exists
    demo_path = Path(__file__).parent / "demo_pipeline.py"
    if demo_path.exists():
        print(f"✅ Demo script found at {demo_path}")
    else:
        print(f"❌ Demo script not found at {demo_path}")
        return False

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\nTo install missing dependencies:")
        print("  poetry install  # if using poetry")
        print("  # or")
        print(f"  pip install {' '.join(missing_deps)}")
        return False

    print("\n✅ All requirements met! You can run the demo:")
    print("  python demo_pipeline.py --help")
    return True

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)
