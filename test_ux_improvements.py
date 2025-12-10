#!/usr/bin/env python3
"""
QA Test for UX Improvements
Tests all changes: LoRA dropdown, multiple schedulers, default negative prompt, logical ordering.
"""

import sys
import ast
from pathlib import Path


def test_imports():
    """Test that all necessary imports are present"""
    print("=" * 60)
    print("TEST 1: Verifying imports")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    required_imports = [
        "import folder_paths",
        "from diffusers.schedulers import FlowMatchEulerDiscreteScheduler",
        "from diffusers.schedulers import (",
        "DPMSolverMultistepScheduler",
        "UniPCMultistepScheduler",
        "EulerDiscreteScheduler",
        "DDIMScheduler",
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


def test_lora_dropdown():
    """Test that LoRA dropdown is properly implemented"""
    print("\n" + "=" * 60)
    print("TEST 2: Verifying LoRA dropdown")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for lora_list initialization
    if 'lora_list = ["[None]", "[Custom Path]"]' not in content:
        print("❌ FAIL: lora_list not initialized correctly")
        return False
    print("✅ PASS: lora_list initialized with [None] and [Custom Path]")

    # Check for folder_paths.get_filename_list("loras")
    if 'folder_paths.get_filename_list("loras")' not in content:
        print("❌ FAIL: Not fetching LoRA list from folder_paths")
        return False
    print("✅ PASS: Fetching LoRA list from folder_paths")

    # Check for lora_selection parameter
    if '"lora_selection"' not in content:
        print("❌ FAIL: lora_selection parameter not found")
        return False
    print("✅ PASS: lora_selection parameter present")

    # Check for lora_custom_path parameter
    if '"lora_custom_path"' not in content:
        print("❌ FAIL: lora_custom_path parameter not found")
        return False
    print("✅ PASS: lora_custom_path parameter present")

    # Check for LoRA path resolution logic
    if 'if lora_selection == "[None]":' not in content:
        print("❌ FAIL: LoRA path resolution logic not found")
        return False
    print("✅ PASS: LoRA path resolution logic present")

    return True


def test_scheduler_list():
    """Test that multiple schedulers are available"""
    print("\n" + "=" * 60)
    print("TEST 3: Verifying scheduler list")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    required_schedulers = [
        "FlowMatchEulerDiscreteScheduler",
        "DPMSolverMultistepScheduler",
        "UniPCMultistepScheduler",
        "EulerDiscreteScheduler",
        "EulerAncestralDiscreteScheduler",
        "DDIMScheduler",
    ]

    # Check scheduler_list definition
    if "scheduler_list = [" not in content:
        print("❌ FAIL: scheduler_list not defined")
        return False
    print("✅ PASS: scheduler_list defined")

    # Check for required schedulers
    missing_schedulers = []
    for scheduler in required_schedulers:
        if f'"{scheduler}"' not in content:
            missing_schedulers.append(scheduler)

    if missing_schedulers:
        print("❌ FAIL: Missing schedulers in list:")
        for sched in missing_schedulers:
            print(f"  - {sched}")
        return False
    print(f"✅ PASS: All {len(required_schedulers)} required schedulers present")

    # Check scheduler_map has all schedulers
    if "scheduler_map = {" not in content:
        print("❌ FAIL: scheduler_map not found")
        return False
    print("✅ PASS: scheduler_map found")

    # Check that all schedulers are in the map
    for scheduler in required_schedulers:
        if f'"{scheduler}": {scheduler}' not in content:
            print(f"❌ FAIL: {scheduler} not in scheduler_map")
            return False
    print("✅ PASS: All schedulers in scheduler_map")

    return True


def test_default_negative_prompt():
    """Test that default negative prompt is set"""
    print("\n" + "=" * 60)
    print("TEST 4: Verifying default negative prompt")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for negative_prompt parameter
    if '"negative_prompt"' not in content:
        print("❌ FAIL: negative_prompt parameter not found")
        return False
    print("✅ PASS: negative_prompt parameter present")

    # Check for default value (should not be empty)
    if '"default": "blurry' not in content:
        print("❌ FAIL: Default negative prompt not set or incorrect")
        return False
    print("✅ PASS: Default negative prompt set correctly")

    # Check it's in required section (not optional)
    # Find negative_prompt position
    neg_prompt_pos = content.find('"negative_prompt"')
    required_section_end = content.find('"optional":')

    if required_section_end == -1 or neg_prompt_pos > required_section_end:
        print("❌ FAIL: negative_prompt not in required section")
        return False
    print("✅ PASS: negative_prompt in required section")

    return True


def test_parameter_ordering():
    """Test that parameters are ordered logically"""
    print("\n" + "=" * 60)
    print("TEST 5: Verifying parameter ordering")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for group comments
    expected_groups = [
        "# GROUP 1: MODEL SELECTION",
        "# GROUP 2: GENERATION PROMPTS",
        "# GROUP 3: GENERATION SETTINGS",
        "# GROUP 4: MODEL CONFIGURATION",
        "# GROUP 5: ENHANCEMENTS",
    ]

    missing_groups = []
    for group in expected_groups:
        if group not in content:
            missing_groups.append(group)

    if missing_groups:
        print("❌ FAIL: Missing logical grouping comments:")
        for group in missing_groups:
            print(f"  - {group}")
        return False
    print(f"✅ PASS: All {len(expected_groups)} logical groups present")

    # Check order of key parameters
    param_positions = {}
    key_params = [
        '"model_selection"',
        '"prompt"',
        '"negative_prompt"',
        '"steps"',
        '"scheduler"',
        '"dtype"',
        '"lora_selection"',
    ]

    for param in key_params:
        pos = content.find(param)
        if pos == -1:
            print(f"❌ FAIL: Parameter {param} not found")
            return False
        param_positions[param] = pos

    # Check logical ordering (model selection before prompts before settings before enhancements)
    if not (param_positions['"model_selection"'] < param_positions['"prompt"'] <
            param_positions['"steps"'] < param_positions['"lora_selection"']):
        print("❌ FAIL: Parameters not in logical order")
        print(f"  Positions: {param_positions}")
        return False
    print("✅ PASS: Parameters in logical order")

    return True


def test_generate_signature():
    """Test that generate() signature matches new parameters"""
    print("\n" + "=" * 60)
    print("TEST 6: Verifying generate() signature")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for new parameters in signature
    if "lora_selection: str" not in content:
        print("❌ FAIL: lora_selection not in generate() signature")
        return False
    print("✅ PASS: lora_selection in generate() signature")

    if "lora_custom_path: str" not in content:
        print("❌ FAIL: lora_custom_path not in generate() signature")
        return False
    print("✅ PASS: lora_custom_path in generate() signature")

    # Check parameter order in signature matches INPUT_TYPES
    sig_params = [
        "negative_prompt",
        "steps",
        "scheduler",
        "lora_selection",
        "lora_custom_path",
    ]

    for param in sig_params:
        if f"{param}:" not in content and f"{param} :" not in content:
            print(f"❌ FAIL: {param} not in generate() signature")
            return False
    print(f"✅ PASS: All {len(sig_params)} parameters in generate() signature")

    return True


def test_syntax():
    """Test Python syntax is valid"""
    print("\n" + "=" * 60)
    print("TEST 7: Verifying Python syntax")
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
    print("QA VALIDATION: UX Improvements")
    print("🔍" * 30 + "\n")

    tests = [
        test_imports,
        test_lora_dropdown,
        test_scheduler_list,
        test_default_negative_prompt,
        test_parameter_ordering,
        test_generate_signature,
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
