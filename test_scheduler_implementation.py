#!/usr/bin/env python3
"""
QA Test for Scheduler Implementation
Tests structure and imports without requiring full dependencies.
"""

import sys
import ast
from pathlib import Path


def test_imports():
    """Test that all necessary imports are present in sampler.py"""
    print("=" * 60)
    print("TEST 1: Verifying imports")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    required_imports = [
        "from diffusers.schedulers import FlowMatchEulerDiscreteScheduler",
        "from diffusers import DiffusionPipeline",
        "from sdnq import SDNQConfig",
    ]

    missing = []
    for imp in required_imports:
        if imp not in content:
            missing.append(imp)

    if missing:
        print("❌ FAIL: Missing imports:")
        for imp in missing:
            print(f"  - {imp}")
        return False

    print("✅ PASS: All required imports present")
    return True


def test_input_types():
    """Test that INPUT_TYPES includes all new parameters"""
    print("\n" + "=" * 60)
    print("TEST 2: Verifying INPUT_TYPES parameters")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for lora_strength range
    if 'min": -5.0' not in content or 'max": 5.0' not in content:
        print("❌ FAIL: lora_strength range not updated to -5.0 to +5.0")
        return False
    print("✅ PASS: lora_strength range is -5.0 to +5.0")

    # Check for scheduler parameter
    if '"scheduler"' not in content:
        print("❌ FAIL: scheduler parameter not added")
        return False
    print("✅ PASS: scheduler parameter present")

    # Check scheduler is in optional section
    if '"scheduler": ([' not in content:
        print("❌ FAIL: scheduler not properly formatted")
        return False
    print("✅ PASS: scheduler properly formatted as dropdown")

    # Check for FlowMatchEulerDiscreteScheduler in dropdown
    if 'FlowMatchEulerDiscreteScheduler' not in content:
        print("❌ FAIL: FlowMatchEulerDiscreteScheduler not in dropdown")
        return False
    print("✅ PASS: FlowMatchEulerDiscreteScheduler in dropdown")

    return True


def test_method_signatures():
    """Test that method signatures are correct"""
    print("\n" + "=" * 60)
    print("TEST 3: Verifying method signatures")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check generate() signature includes scheduler
    if 'scheduler: str' not in content:
        print("❌ FAIL: generate() method doesn't include scheduler parameter")
        return False
    print("✅ PASS: generate() signature includes scheduler parameter")

    # Check swap_scheduler method exists
    if 'def swap_scheduler(self' not in content:
        print("❌ FAIL: swap_scheduler() method not found")
        return False
    print("✅ PASS: swap_scheduler() method exists")

    # Check __init__ tracks current_scheduler
    if 'self.current_scheduler' not in content:
        print("❌ FAIL: __init__ doesn't track current_scheduler")
        return False
    print("✅ PASS: __init__ tracks current_scheduler")

    return True


def test_logic_flow():
    """Test that scheduler swap logic is in the right place"""
    print("\n" + "=" * 60)
    print("TEST 4: Verifying logic flow")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check scheduler swap is called in generate()
    if 'self.swap_scheduler(self.pipeline, scheduler)' not in content:
        print("❌ FAIL: swap_scheduler() not called in generate()")
        return False
    print("✅ PASS: swap_scheduler() called in generate()")

    # Check scheduler cache is cleared when pipeline reloads
    if 'self.current_scheduler = None' not in content:
        print("❌ FAIL: scheduler cache not cleared on pipeline reload")
        return False
    print("✅ PASS: scheduler cache cleared on pipeline reload")

    # Check scheduler_map exists
    if 'scheduler_map = {' not in content:
        print("❌ FAIL: scheduler_map not found in swap_scheduler()")
        return False
    print("✅ PASS: scheduler_map exists")

    # Check from_config pattern is used
    if 'from_config(pipeline.scheduler.config)' not in content:
        print("❌ FAIL: from_config pattern not used")
        return False
    print("✅ PASS: from_config pattern used correctly")

    return True


def test_syntax():
    """Test Python syntax is valid"""
    print("\n" + "=" * 60)
    print("TEST 5: Verifying Python syntax")
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
    print("QA VALIDATION: Scheduler Implementation")
    print("🔍" * 30 + "\n")

    tests = [
        test_imports,
        test_input_types,
        test_method_signatures,
        test_logic_flow,
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
