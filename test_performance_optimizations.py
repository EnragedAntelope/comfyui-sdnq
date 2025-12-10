#!/usr/bin/env python3
"""
QA Test for Performance Optimizations Implementation
Tests all changes: torch.compile, xFormers, VAE tiling parameters and logic.
"""

import sys
import ast
from pathlib import Path


def test_performance_parameters():
    """Test that performance optimization parameters are in INPUT_TYPES"""
    print("=" * 60)
    print("TEST 1: Verifying performance optimization parameters")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for use_torch_compile parameter
    if '"use_torch_compile"' not in content:
        print("❌ FAIL: use_torch_compile parameter not found")
        return False
    print("✅ PASS: use_torch_compile parameter present")

    # Check for use_xformers parameter
    if '"use_xformers"' not in content:
        print("❌ FAIL: use_xformers parameter not found")
        return False
    print("✅ PASS: use_xformers parameter present")

    # Check for enable_vae_tiling parameter
    if '"enable_vae_tiling"' not in content:
        print("❌ FAIL: enable_vae_tiling parameter not found")
        return False
    print("✅ PASS: enable_vae_tiling parameter present")

    return True


def test_parameter_defaults():
    """Test that performance parameters have correct defaults"""
    print("\n" + "=" * 60)
    print("TEST 2: Verifying parameter defaults")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check use_torch_compile default (should be False for backward compat)
    if '"default": False' not in content.split('"use_torch_compile"')[1].split('}')[0]:
        print("❌ FAIL: use_torch_compile default not False")
        return False
    print("✅ PASS: use_torch_compile default is False")

    # Check use_xformers default (should be False for safety/compatibility)
    if '"default": False' not in content.split('"use_xformers"')[1].split('}')[0]:
        print("❌ FAIL: use_xformers default not False")
        return False
    print("✅ PASS: use_xformers default is False (safe default)")

    # Check enable_vae_tiling default (should be False, only for large images)
    if '"default": False' not in content.split('"enable_vae_tiling"')[1].split('}')[0]:
        print("❌ FAIL: enable_vae_tiling default not False")
        return False
    print("✅ PASS: enable_vae_tiling default is False")

    return True


def test_tooltips():
    """Test that performance parameters have good tooltip text"""
    print("\n" + "=" * 60)
    print("TEST 3: Verifying tooltip quality")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check torch.compile tooltip mentions speedup and compilation overhead
    torch_compile_section = content.split('"use_torch_compile"')[1].split('}')[0]
    if "speedup" not in torch_compile_section.lower() or "compilation" not in torch_compile_section.lower():
        print("❌ FAIL: use_torch_compile tooltip missing key information")
        return False
    print("✅ PASS: use_torch_compile tooltip has speedup and compilation info")

    # Check xformers tooltip mentions fallback
    xformers_section = content.split('"use_xformers"')[1].split('}')[0]
    if "fallback" not in xformers_section.lower() or "sdpa" not in xformers_section.lower():
        print("❌ FAIL: use_xformers tooltip missing fallback information")
        return False
    print("✅ PASS: use_xformers tooltip mentions graceful fallback to SDPA")

    # Check VAE tiling tooltip mentions large images
    vae_section = content.split('"enable_vae_tiling"')[1].split('}')[0]
    if "large" not in vae_section.lower():
        print("❌ FAIL: enable_vae_tiling tooltip missing large image info")
        return False
    print("✅ PASS: enable_vae_tiling tooltip mentions large images")

    return True


def test_method_existence():
    """Test that apply_performance_optimizations() method exists"""
    print("\n" + "=" * 60)
    print("TEST 4: Verifying apply_performance_optimizations() method")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check method exists
    if "def apply_performance_optimizations(self" not in content:
        print("❌ FAIL: apply_performance_optimizations() method not found")
        return False
    print("✅ PASS: apply_performance_optimizations() method exists")

    # Check method has correct parameters
    method_section = content.split("def apply_performance_optimizations(self")[1].split("):")[0]
    if "use_torch_compile" not in method_section:
        print("❌ FAIL: Method missing use_torch_compile parameter")
        return False
    if "use_xformers" not in method_section:
        print("❌ FAIL: Method missing use_xformers parameter")
        return False
    if "enable_vae_tiling" not in method_section:
        print("❌ FAIL: Method missing enable_vae_tiling parameter")
        return False
    print("✅ PASS: Method has all required parameters")

    return True


def test_optimization_logic():
    """Test that optimization logic is implemented correctly"""
    print("\n" + "=" * 60)
    print("TEST 5: Verifying optimization logic")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for xFormers logic
    if "enable_xformers_memory_efficient_attention()" not in content:
        print("❌ FAIL: xFormers enable call not found")
        return False
    print("✅ PASS: xFormers enable call present")

    # Check for torch.compile logic
    if "torch.compile(" not in content:
        print("❌ FAIL: torch.compile call not found")
        return False
    print("✅ PASS: torch.compile call present")

    # Check for mode="max-autotune"
    if 'mode="max-autotune"' not in content:
        print("❌ FAIL: torch.compile not using max-autotune mode")
        return False
    print("✅ PASS: torch.compile using max-autotune mode (best for latency)")

    # Check for channels_last memory format
    if "memory_format=torch.channels_last" not in content:
        print("❌ FAIL: channels_last memory format not set")
        return False
    print("✅ PASS: channels_last memory format set")

    # Check for VAE tiling logic
    if "enable_vae_tiling()" not in content:
        print("❌ FAIL: VAE tiling enable call not found")
        return False
    print("✅ PASS: VAE tiling enable call present")

    # Check for try/except error handling
    optimization_section = content.split("def apply_performance_optimizations")[1].split("def load_lora")[0]
    try_count = optimization_section.count("try:")
    except_count = optimization_section.count("except Exception")
    if try_count < 3 or except_count < 3:
        print(f"❌ FAIL: Insufficient error handling (found {try_count} try blocks, {except_count} except blocks)")
        return False
    print(f"✅ PASS: Proper error handling ({try_count} try/except blocks)")

    return True


def test_load_pipeline_signature():
    """Test that load_pipeline() signature includes performance parameters"""
    print("\n" + "=" * 60)
    print("TEST 6: Verifying load_pipeline() signature")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check signature includes performance parameters
    signature_section = content.split("def load_pipeline(self")[1].split("):")[0]

    if "use_torch_compile" not in signature_section:
        print("❌ FAIL: load_pipeline() missing use_torch_compile parameter")
        return False
    if "use_xformers" not in signature_section:
        print("❌ FAIL: load_pipeline() missing use_xformers parameter")
        return False
    if "enable_vae_tiling" not in signature_section:
        print("❌ FAIL: load_pipeline() missing enable_vae_tiling parameter")
        return False
    print("✅ PASS: load_pipeline() has all performance optimization parameters")

    # Check that apply_performance_optimizations is called
    load_pipeline_body = content.split("def load_pipeline(self")[1].split("def apply_performance_optimizations")[0]
    if "self.apply_performance_optimizations(" not in load_pipeline_body:
        print("❌ FAIL: load_pipeline() doesn't call apply_performance_optimizations()")
        return False
    print("✅ PASS: load_pipeline() calls apply_performance_optimizations()")

    return True


def test_generate_signature():
    """Test that generate() signature includes performance parameters"""
    print("\n" + "=" * 60)
    print("TEST 7: Verifying generate() signature")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check signature includes performance parameters
    signature_section = content.split("def generate(self")[1].split("):")[0]

    if "use_torch_compile" not in signature_section:
        print("❌ FAIL: generate() missing use_torch_compile parameter")
        return False
    if "use_xformers" not in signature_section:
        print("❌ FAIL: generate() missing use_xformers parameter")
        return False
    if "enable_vae_tiling" not in signature_section:
        print("❌ FAIL: generate() missing enable_vae_tiling parameter")
        return False
    print("✅ PASS: generate() has all performance optimization parameters")

    # Check that parameters are passed to load_pipeline()
    generate_body = content.split("def generate(self")[1].split("def check_interrupted")[0]
    if "self.load_pipeline(" not in generate_body:
        print("❌ FAIL: generate() doesn't call load_pipeline()")
        return False

    load_pipeline_call = generate_body.split("self.load_pipeline(")[1].split(")")[0]
    if "use_torch_compile=" not in load_pipeline_call:
        print("❌ FAIL: generate() doesn't pass use_torch_compile to load_pipeline()")
        return False
    print("✅ PASS: generate() passes performance parameters to load_pipeline()")

    return True


def test_cache_tracking():
    """Test that cache tracking includes optimization settings"""
    print("\n" + "=" * 60)
    print("TEST 8: Verifying cache tracking")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check __init__ tracks optimization settings
    init_section = content.split("def __init__(self)")[1].split("@classmethod")[0]

    if "self.current_use_torch_compile" not in init_section:
        print("❌ FAIL: __init__ doesn't track current_use_torch_compile")
        return False
    if "self.current_use_xformers" not in init_section:
        print("❌ FAIL: __init__ doesn't track current_use_xformers")
        return False
    if "self.current_enable_vae_tiling" not in init_section:
        print("❌ FAIL: __init__ doesn't track current_enable_vae_tiling")
        return False
    print("✅ PASS: __init__ tracks all optimization settings")

    # Check cache invalidation logic includes optimization settings
    generate_body = content.split("def generate(self")[1]
    cache_check_section = generate_body.split("if (self.pipeline is None")[1].split("):")[0]

    if "self.current_use_torch_compile != use_torch_compile" not in cache_check_section:
        print("❌ FAIL: Cache invalidation doesn't check use_torch_compile")
        return False
    if "self.current_use_xformers != use_xformers" not in cache_check_section:
        print("❌ FAIL: Cache invalidation doesn't check use_xformers")
        return False
    if "self.current_enable_vae_tiling != enable_vae_tiling" not in cache_check_section:
        print("❌ FAIL: Cache invalidation doesn't check enable_vae_tiling")
        return False
    print("✅ PASS: Cache invalidation checks all optimization settings")

    # Check cache is updated after pipeline load
    cache_update_section = generate_body.split("self.current_model_path = model_path")[1].split("# Step 2.5")[0]

    if "self.current_use_torch_compile = use_torch_compile" not in cache_update_section:
        print("❌ FAIL: Cache not updated with use_torch_compile")
        return False
    if "self.current_use_xformers = use_xformers" not in cache_update_section:
        print("❌ FAIL: Cache not updated with use_xformers")
        return False
    if "self.current_enable_vae_tiling = enable_vae_tiling" not in cache_update_section:
        print("❌ FAIL: Cache not updated with enable_vae_tiling")
        return False
    print("✅ PASS: Cache properly updated with optimization settings")

    return True


def test_memory_mode_compatibility():
    """Test that torch.compile checks memory_mode compatibility"""
    print("\n" + "=" * 60)
    print("TEST 9: Verifying memory_mode compatibility check")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check that apply_performance_optimizations accepts memory_mode parameter
    optimization_method = content.split("def apply_performance_optimizations(self")[1].split("):")[0]
    if "memory_mode" not in optimization_method:
        print("❌ FAIL: apply_performance_optimizations doesn't accept memory_mode parameter")
        return False
    print("✅ PASS: apply_performance_optimizations accepts memory_mode parameter")

    # Check that torch.compile has memory mode check
    if 'if memory_mode != "gpu":' not in content:
        print("❌ FAIL: torch.compile doesn't check memory_mode compatibility")
        return False
    print("✅ PASS: torch.compile checks if memory_mode == 'gpu'")

    # Check that warning is printed for incompatible mode
    if "torch.compile is incompatible with memory_mode" not in content:
        print("❌ FAIL: No warning for incompatible memory_mode")
        return False
    print("✅ PASS: Warning printed for incompatible memory modes")

    # Check that memory_mode is passed to apply_performance_optimizations
    load_pipeline_body = content.split("def load_pipeline(self")[1].split("def apply_performance_optimizations")[0]
    if "memory_mode=memory_mode" not in load_pipeline_body:
        print("❌ FAIL: memory_mode not passed to apply_performance_optimizations()")
        return False
    print("✅ PASS: memory_mode passed to apply_performance_optimizations()")

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
    print("QA VALIDATION: Performance Optimizations")
    print("🔍" * 30 + "\n")

    tests = [
        test_performance_parameters,
        test_parameter_defaults,
        test_tooltips,
        test_method_existence,
        test_optimization_logic,
        test_load_pipeline_signature,
        test_generate_signature,
        test_cache_tracking,
        test_memory_mode_compatibility,
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
        print("\n🎉 ALL TESTS PASSED - Implementation is ready!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed - Fix issues before committing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
