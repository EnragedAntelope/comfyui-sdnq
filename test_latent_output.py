#!/usr/bin/env python3
"""
QA Test for LATENT Output Feature

Validates that the SDNQ Sampler correctly outputs both IMAGE and LATENT
without breaking existing functionality.
"""

import sys
import ast
from pathlib import Path


def test_return_types():
    """Test that RETURN_TYPES includes both IMAGE and LATENT"""
    print("=" * 60)
    print("TEST 1: Verifying RETURN_TYPES")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check RETURN_TYPES includes IMAGE and LATENT
    if 'RETURN_TYPES = ("IMAGE", "LATENT")' not in content:
        print("❌ FAIL: RETURN_TYPES doesn't include both IMAGE and LATENT")
        return False
    print("✅ PASS: RETURN_TYPES = ('IMAGE', 'LATENT')")

    # Check RETURN_NAMES
    if 'RETURN_NAMES = ("image", "latent")' not in content:
        print("❌ FAIL: RETURN_NAMES doesn't match")
        return False
    print("✅ PASS: RETURN_NAMES = ('image', 'latent')")

    return True


def test_generate_image_signature():
    """Test that generate_image returns Tuple[Image, Tensor]"""
    print("\n" + "=" * 60)
    print("TEST 2: Verifying generate_image() signature")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check return type hint
    if "-> Tuple[Image.Image, torch.Tensor]:" not in content:
        print("❌ FAIL: generate_image() doesn't have correct return type hint")
        return False
    print("✅ PASS: generate_image() returns Tuple[Image.Image, torch.Tensor]")

    # Check docstring mentions both returns
    generate_section = content.split("def generate_image(self")[1].split("def pil_to_comfy_tensor")[0]
    if "Tuple of (PIL Image object, latent tensor)" not in generate_section:
        print("❌ FAIL: generate_image() docstring doesn't document both returns")
        return False
    print("✅ PASS: generate_image() docstring documents both returns")

    return True


def test_output_type_latent():
    """Test that pipeline is called with output_type='latent'"""
    print("\n" + "=" * 60)
    print("TEST 3: Verifying output_type='latent' usage")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check output_type="latent" is set
    if '"output_type": "latent"' not in content:
        print("❌ FAIL: pipeline not called with output_type='latent'")
        return False
    print("✅ PASS: pipeline called with output_type='latent'")

    # Check comment explains why
    generate_section = content.split("def generate_image(self")[1].split("def pil_to_comfy_tensor")[0]
    if "Get latents before VAE decode" not in generate_section:
        print("❌ FAIL: Missing explanatory comment for output_type='latent'")
        return False
    print("✅ PASS: Comment explains latent extraction")

    return True


def test_manual_vae_decode():
    """Test that VAE decode is done manually"""
    print("\n" + "=" * 60)
    print("TEST 4: Verifying manual VAE decode")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    generate_section = content.split("def generate_image(self")[1].split("def pil_to_comfy_tensor")[0]

    # Check for VAE scaling factor usage
    if "vae_scale_factor = pipeline.vae.config.scaling_factor" not in generate_section:
        print("❌ FAIL: Not using VAE scaling factor")
        return False
    print("✅ PASS: Uses VAE scaling factor")

    # Check for manual decode call
    if "pipeline.vae.decode(latents_batch / vae_scale_factor)" not in generate_section:
        print("❌ FAIL: Not manually calling VAE decode")
        return False
    print("✅ PASS: Manually calls pipeline.vae.decode()")

    # Check for proper tensor conversion [-1, 1] -> [0, 1]
    if "(decoded / 2 + 0.5).clamp(0, 1)" not in generate_section:
        print("❌ FAIL: Not properly normalizing decoded tensor")
        return False
    print("✅ PASS: Properly normalizes decoded tensor [-1, 1] -> [0, 1]")

    # Check for PIL Image creation
    if "Image.fromarray(decoded)" not in generate_section:
        print("❌ FAIL: Not creating PIL Image from decoded tensor")
        return False
    print("✅ PASS: Creates PIL Image from decoded tensor")

    return True


def test_return_both_outputs():
    """Test that generate_image returns both image and latents"""
    print("\n" + "=" * 60)
    print("TEST 5: Verifying generate_image returns both outputs")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    generate_section = content.split("def generate_image(self")[1].split("def pil_to_comfy_tensor")[0]

    # Check for return statement
    if "return image, latents" not in generate_section:
        print("❌ FAIL: generate_image() doesn't return both image and latents")
        return False
    print("✅ PASS: generate_image() returns (image, latents)")

    return True


def test_latents_to_comfy_format():
    """Test that latents_to_comfy_format helper exists and is correct"""
    print("\n" + "=" * 60)
    print("TEST 6: Verifying latents_to_comfy_format() helper")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check method exists
    if "def latents_to_comfy_format(self, latents: torch.Tensor)" not in content:
        print("❌ FAIL: latents_to_comfy_format() method not found")
        return False
    print("✅ PASS: latents_to_comfy_format() method exists")

    # Check return type hint
    if "-> Dict[str, torch.Tensor]:" not in content:
        print("❌ FAIL: latents_to_comfy_format() missing Dict return type")
        return False
    print("✅ PASS: Returns Dict[str, torch.Tensor]")

    helper_section = content.split("def latents_to_comfy_format(self")[1].split("def generate(self")[0]

    # Check returns dict with "samples" key
    if 'return {"samples": latents}' not in helper_section:
        print("❌ FAIL: Doesn't return {'samples': latents}")
        return False
    print("✅ PASS: Returns {'samples': latents}")

    # Check moves to CPU
    if "latents.cpu()" not in helper_section:
        print("❌ FAIL: Doesn't move latents to CPU")
        return False
    print("✅ PASS: Moves latents to CPU")

    # Check converts to float32
    if ".float()" not in helper_section:
        print("❌ FAIL: Doesn't convert to float32")
        return False
    print("✅ PASS: Converts to float32")

    return True


def test_generate_method_integration():
    """Test that generate() method properly integrates latent output"""
    print("\n" + "=" * 60)
    print("TEST 7: Verifying generate() method integration")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    generate_section = content.split("def generate(self, model_selection")[1].split("except InterruptedError")[0]

    # Check unpacks both from generate_image()
    if "pil_image, latents = self.generate_image(" not in generate_section:
        print("❌ FAIL: generate() doesn't unpack both outputs from generate_image()")
        return False
    print("✅ PASS: Unpacks (pil_image, latents) from generate_image()")

    # Check converts image to ComfyUI format
    if "comfy_image = self.pil_to_comfy_tensor(pil_image)" not in generate_section:
        print("❌ FAIL: Doesn't convert PIL image to ComfyUI tensor")
        return False
    print("✅ PASS: Converts PIL image to ComfyUI IMAGE tensor")

    # Check converts latents to ComfyUI format
    if "comfy_latent = self.latents_to_comfy_format(latents)" not in generate_section:
        print("❌ FAIL: Doesn't convert latents to ComfyUI LATENT format")
        return False
    print("✅ PASS: Converts latents to ComfyUI LATENT format")

    # Check returns both
    if "return (comfy_image, comfy_latent)" not in generate_section:
        print("❌ FAIL: Doesn't return both comfy_image and comfy_latent")
        return False
    print("✅ PASS: Returns (comfy_image, comfy_latent)")

    return True


def test_docstring_updates():
    """Test that docstrings are updated"""
    print("\n" + "=" * 60)
    print("TEST 8: Verifying docstring updates")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check generate() docstring mentions both outputs
    generate_section = content.split("def generate(self, model_selection")[1].split("print(f\"\\n")[0]
    if "Tuple containing (IMAGE, LATENT)" not in generate_section:
        print("❌ FAIL: generate() docstring doesn't document both outputs")
        return False
    print("✅ PASS: generate() docstring documents (IMAGE, LATENT) tuple")

    # Check format specifications
    if "[1, H, W, 3]" not in generate_section:
        print("❌ FAIL: Missing IMAGE format specification")
        return False
    if "[1, C, H/8, W/8]" not in generate_section:
        print("❌ FAIL: Missing LATENT format specification")
        return False
    print("✅ PASS: Docstring specifies both formats correctly")

    return True


def test_no_breaking_changes():
    """Test that existing code paths aren't broken"""
    print("\n" + "=" * 60)
    print("TEST 9: Verifying no breaking changes")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Verify INPUT_TYPES still exists
    if "@classmethod\n    def INPUT_TYPES(cls):" not in content:
        print("❌ FAIL: INPUT_TYPES method missing")
        return False
    print("✅ PASS: INPUT_TYPES method intact")

    # Verify FUNCTION still set
    if 'FUNCTION = "generate"' not in content:
        print("❌ FAIL: FUNCTION attribute missing")
        return False
    print("✅ PASS: FUNCTION = 'generate' intact")

    # Verify CATEGORY still set
    if 'CATEGORY = "sampling/SDNQ"' not in content:
        print("❌ FAIL: CATEGORY attribute missing")
        return False
    print("✅ PASS: CATEGORY intact")

    # Verify node registration
    if 'NODE_CLASS_MAPPINGS = {\n    "SDNQSampler": SDNQSampler,' not in content:
        print("❌ FAIL: NODE_CLASS_MAPPINGS missing")
        return False
    print("✅ PASS: NODE_CLASS_MAPPINGS intact")

    return True


def test_syntax():
    """Test Python syntax is valid"""
    print("\n" + "=" * 60)
    print("TEST 10: Verifying Python syntax")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    try:
        ast.parse(content)
        print("✅ PASS: Python syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ FAIL: Syntax error at line {e.lineno}: {e.msg}")
        print(f"   {e.text}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🔍" * 30)
    print("QA VALIDATION: LATENT Output Feature")
    print("🔍" * 30 + "\n")

    tests = [
        test_return_types,
        test_generate_image_signature,
        test_output_type_latent,
        test_manual_vae_decode,
        test_return_both_outputs,
        test_latents_to_comfy_format,
        test_generate_method_integration,
        test_docstring_updates,
        test_no_breaking_changes,
        test_syntax,
    ]

    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"❌ FAIL: Test {test_func.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - LATENT output implementation is ready!")
        print("\nFeature summary:")
        print("- IMAGE output: PIL image decoded from latents (existing functionality)")
        print("- LATENT output: Raw latents before VAE decode (new functionality)")
        print("- Users can choose which output to use in their workflow")
        print("- No breaking changes to existing functionality")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed - Fix issues before committing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
