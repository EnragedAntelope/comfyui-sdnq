"""
ComfyUI Manager Install Hook

This script is called by ComfyUI Manager during installation.
It ensures all dependencies are properly installed.
"""

import subprocess
import sys
import os


def install():
    """Install required dependencies for ComfyUI-SDNQ"""
    print("\n" + "="*60)
    print("Installing ComfyUI-SDNQ Dependencies")
    print("="*60 + "\n")

    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if not os.path.exists(requirements_file):
        print("Error: requirements.txt not found!")
        return False

    try:
        print("Installing from requirements.txt...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file
        ])

        print("\n" + "="*60)
        print("[OK] ComfyUI-SDNQ installation complete!")
        print("="*60 + "\n")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Installation failed: {e}")
        print("\nTry manual installation:")
        print(f"  pip install -r {requirements_file}")
        return False


if __name__ == "__main__":
    install()
