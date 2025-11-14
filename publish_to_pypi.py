#!/usr/bin/env python3
"""
Script to build and publish Quantum Conduit to PyPI.

This script will:
1. Clean previous builds
2. Build source distribution and wheel
3. Check the package
4. Optionally upload to PyPI (or TestPyPI for testing)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def clean_build_artifacts():
    """Remove previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    dirs_to_remove = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_remove:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   Removed {path}")
            elif path.is_file():
                path.unlink()
                print(f"   Removed {path}")


def build_package():
    """Build the source distribution and wheel."""
    print("\n📦 Building package...")
    run_command([sys.executable, "-m", "build"])


def check_package():
    """Check the package with twine."""
    print("\n✅ Checking package...")
    run_command([sys.executable, "-m", "twine", "check", "dist/*"])


def upload_to_pypi(test: bool = False, skip_existing: bool = False):
    """Upload package to PyPI or TestPyPI."""
    repository = "testpypi" if test else "pypi"
    repo_name = "TestPyPI" if test else "PyPI"
    
    print(f"\n🚀 Uploading to {repo_name}...")
    
    cmd = [sys.executable, "-m", "twine", "upload", "dist/*"]
    cmd.extend(["--repository", repository])
    
    if skip_existing:
        cmd.append("--skip-existing")
    
    # Check for credentials
    username = os.getenv("PYPI_USERNAME")
    password = os.getenv("PYPI_PASSWORD")
    
    if not username or not password:
        print(f"\n⚠️  PYPI_USERNAME and PYPI_PASSWORD environment variables not set.")
        print(f"   You can either:")
        print(f"   1. Set environment variables: export PYPI_USERNAME=... PYPI_PASSWORD=...")
        print(f"   2. Use API tokens: export PYPI_USERNAME=__token__ PYPI_PASSWORD=pypi-...")
        print(f"   3. Enter credentials when prompted")
        print(f"\n   For {repo_name}:")
        if test:
            print(f"   - TestPyPI: https://test.pypi.org/manage/account/token/")
        else:
            print(f"   - PyPI: https://pypi.org/manage/account/token/")
        response = input(f"\n   Continue with manual entry? (y/n): ").strip().lower()
        if response != 'y':
            print("   Cancelled.")
            return False
    
    run_command(cmd)
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and publish Quantum Conduit to PyPI")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Upload to TestPyPI instead of PyPI"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist on PyPI"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the package, don't upload"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean previous build artifacts"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Quantum Conduit - PyPI Publishing Script")
    print("="*70)
    
    # Check required tools
    print("\n🔍 Checking required tools...")
    try:
        import build
        print("   ✓ build module found")
    except ImportError:
        print("   ✗ build module not found. Install with: pip install build")
        sys.exit(1)
    
    if not args.build_only:
        try:
            import twine
            print("   ✓ twine module found")
        except ImportError:
            print("   ✗ twine module not found. Install with: pip install twine")
            sys.exit(1)
    else:
        print("   ℹ️  twine not required for build-only mode")
    
    # Clean
    if not args.no_clean:
        clean_build_artifacts()
    
    # Build
    build_package()
    
    # Check (requires twine)
    if not args.build_only:
        check_package()
    else:
        print("\n⚠️  Skipping package check (twine required)")
    
    # Upload
    if not args.build_only:
        if args.test:
            print("\n⚠️  Uploading to TestPyPI (for testing)")
            print("   Install from TestPyPI with:")
            print("   pip install --index-url https://test.pypi.org/simple/ qconduit")
        else:
            print("\n⚠️  Uploading to PyPI (PRODUCTION)")
            confirm = input("   Are you sure you want to publish to PyPI? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("   Cancelled.")
                return
        
        upload_to_pypi(test=args.test, skip_existing=args.skip_existing)
        print("\n✅ Done!")
        
        if args.test:
            print("\n📝 To install from TestPyPI:")
            print("   pip install --index-url https://test.pypi.org/simple/ qconduit")
        else:
            print("\n📝 To install from PyPI:")
            print("   pip install qconduit")
    else:
        print("\n✅ Build complete! Package is in the 'dist' directory.")
        print("   To upload manually, run:")
        if args.test:
            print("   twine upload --repository testpypi dist/*")
        else:
            print("   twine upload dist/*")


if __name__ == "__main__":
    main()

