"""
ComfyUI Manager Install Hook

This script is called by ComfyUI Manager during installation.
It verifies that all required dependencies are available.

Dependencies are declared in pyproject.toml and requirements.txt.
ComfyUI Manager and the Comfy Registry handle installation from
pyproject.toml automatically.
"""

import importlib
import os


# Map of package import names to pip install names
REQUIRED_PACKAGES = {
    "sdnq": "sdnq>=0.1.0",
    "diffusers": "diffusers>=0.36.0",
    "huggingface_hub": "huggingface-hub>=0.20.0",
    "safetensors": "safetensors>=0.4.0",
    "accelerate": "accelerate>=0.25.0",
    "hf_xet": "hf-xet>=1.2.0",
}


def install():
    """Verify required dependencies for ComfyUI-SDNQ are installed."""
    print("\n" + "=" * 60)
    print("ComfyUI-SDNQ Dependency Check")
    print("=" * 60 + "\n")

    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
        print("[WARNING] Missing dependencies detected:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print(f"  pip install -r {requirements_file}")
        print()
        return False

    print("[OK] All ComfyUI-SDNQ dependencies are installed.")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    install()
