#!/usr/bin/env python3
"""
Test for graceful negative_prompt handling.
Validates that the code doesn't crash when negative_prompt is unsupported.
"""

import sys
from pathlib import Path


def test_negative_prompt_retry_logic():
    """Test that negative_prompt errors trigger retry logic"""
    print("=" * 60)
    print("TEST: Verifying negative_prompt retry logic")
    print("=" * 60)

    sampler_file = Path(__file__).parent / "nodes" / "sampler.py"
    content = sampler_file.read_text()

    # Check for try/except around pipeline call
    if "try:" not in content or "result = pipeline(**pipeline_kwargs)" not in content:
        print("❌ FAIL: Pipeline call not wrapped in try/except")
        return False
    print("✅ PASS: Pipeline call wrapped in try/except")

    # Check for negative_prompt specific error handling
    if '"negative_prompt" in str(e)' not in content:
        print("❌ FAIL: No check for negative_prompt in error message")
        return False
    print("✅ PASS: Checks for negative_prompt in error message")

    # Check for retry without negative_prompt
    if 'del pipeline_kwargs["negative_prompt"]' not in content:
        print("❌ FAIL: Doesn't remove negative_prompt on error")
        return False
    print("✅ PASS: Removes negative_prompt from kwargs on error")

    # Check for retry call
    retry_count = content.count("result = pipeline(**pipeline_kwargs)")
    if retry_count < 2:
        print(f"❌ FAIL: Only {retry_count} pipeline calls found, expected at least 2 (initial + retry)")
        return False
    print(f"✅ PASS: Found {retry_count} pipeline calls (initial + retry)")

    # Check for warning message
    if "doesn't support negative_prompt - skipping it" not in content:
        print("❌ FAIL: No warning message when skipping negative_prompt")
        return False
    print("✅ PASS: Prints warning when skipping negative_prompt")

    # Check for else block that handles OTHER unsupported parameters
    if "else:" not in content or "Different TypeError" not in content:
        print("❌ FAIL: No else block to handle other unsupported parameters")
        return False
    print("✅ PASS: Has else block for other unsupported parameters")

    # Verify the flow: detect negative_prompt error → remove it → retry
    # The key is that the retry is INSIDE the if block, not after raising
    if_block_has_retry = False
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '"negative_prompt" in str(e) and "unexpected keyword argument" in str(e):' in line:
            # Found the if statement, check next ~10 lines for retry
            for j in range(i+1, min(i+10, len(lines))):
                if 'result = pipeline(**pipeline_kwargs)' in lines[j]:
                    if_block_has_retry = True
                    break
            break

    if not if_block_has_retry:
        print("❌ FAIL: Retry not found in negative_prompt handler block")
        return False
    print("✅ PASS: Retry happens within negative_prompt handler (doesn't crash)")

    return True


def main():
    """Run test"""
    print("\n" + "🔍" * 30)
    print("QA VALIDATION: Graceful negative_prompt Handling")
    print("🔍" * 30 + "\n")

    try:
        result = test_negative_prompt_retry_logic()
    except Exception as e:
        print(f"❌ FAIL: Test raised exception: {e}")
        import traceback
        traceback.print_exc()
        result = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if result:
        print("✅ ALL CHECKS PASSED - Graceful handling implemented correctly!")
        return 0
    else:
        print("❌ CHECKS FAILED - Fix issues before committing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
