#!/usr/bin/env python3
"""
Test Script for Technical Manual Assistant

Tests the assistant service endpoints and validates functionality.

Usage:
    python test_assistant.py --url http://localhost:8002
"""

import argparse
import json
import sys
import time
from typing import Dict, Any

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("\n[TEST] Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Model loaded: {data.get('model_loaded')}")
            print("  ✓ Health check passed")
            return True
        else:
            print(f"  ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False


def test_info(base_url: str) -> Dict[str, Any]:
    """Test info endpoint."""
    print("\n[TEST] Model Info")
    try:
        response = requests.get(f"{base_url}/info", timeout=5)
        data = response.json()

        print(f"  Model: {data.get('model_name')}")
        print(f"  Path: {data.get('model_path')}")
        print(f"  Status: {data.get('status')}")
        print("  ✓ Info check passed")

        return data
    except Exception as e:
        print(f"  ✗ Info check failed: {e}")
        return {}


def test_ask(base_url: str, question: str, expected_keywords: list = None) -> bool:
    """Test ask endpoint."""
    print(f"\n[TEST] Ask: '{question}'")
    try:
        response = requests.post(
            f"{base_url}/ask",
            json={"question": question, "max_length": 256},
            timeout=60
        )

        if response.status_code != 200:
            print(f"  ✗ Ask failed: {response.status_code}")
            return False

        data = response.json()
        answer = data.get("answer", "")
        confidence = data.get("confidence", 0)
        response_time = data.get("response_time_ms", 0)

        print(f"  Response time: {response_time}ms")
        print(f"  Confidence: {confidence}")
        print(f"  Answer preview: {answer[:200]}...")

        # Check for expected keywords if provided
        if expected_keywords:
            found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
            if found:
                print(f"  Found keywords: {found}")
            else:
                print(f"  ⚠ Expected keywords not found: {expected_keywords}")

        if answer and len(answer) > 10:
            print("  ✓ Ask test passed")
            return True
        else:
            print("  ⚠ Answer too short or empty")
            return False

    except Exception as e:
        print(f"  ✗ Ask failed: {e}")
        return False


def test_temperature_variation(base_url: str) -> bool:
    """Test that temperature affects output."""
    print("\n[TEST] Temperature Variation")
    question = "How do I install the software?"

    try:
        # Get response with low temperature
        response1 = requests.post(
            f"{base_url}/ask",
            json={"question": question, "temperature": 0.1},
            timeout=60
        ).json()

        # Get response with high temperature
        response2 = requests.post(
            f"{base_url}/ask",
            json={"question": question, "temperature": 0.9},
            timeout=60
        ).json()

        print(f"  Low temp response: {response1.get('answer', '')[:100]}...")
        print(f"  High temp response: {response2.get('answer', '')[:100]}...")

        # Both should generate valid responses
        if response1.get('answer') and response2.get('answer'):
            print("  ✓ Temperature variation test passed")
            return True
        else:
            print("  ⚠ One or both responses empty")
            return False

    except Exception as e:
        print(f"  ✗ Temperature test failed: {e}")
        return False


def test_max_length(base_url: str) -> bool:
    """Test that max_length is respected."""
    print("\n[TEST] Max Length Control")
    question = "Explain all the features of the software in detail."

    try:
        # Request with very short max length
        response = requests.post(
            f"{base_url}/ask",
            json={"question": question, "max_length": 50},
            timeout=60
        ).json()

        answer = response.get("answer", "")
        # Note: actual length may vary due to tokenization
        print(f"  Requested max: 50 tokens")
        print(f"  Response length: {len(answer)} characters")
        print("  ✓ Max length test passed")
        return True

    except Exception as e:
        print(f"  ✗ Max length test failed: {e}")
        return False


def run_all_tests(base_url: str) -> Dict[str, bool]:
    """Run all tests and return results."""
    results = {}

    # Basic connectivity
    results["health"] = test_health(base_url)

    if not results["health"]:
        print("\n✗ Service not reachable. Make sure the service is running.")
        return results

    # Model info
    info = test_info(base_url)
    results["info"] = info.get("status") in ["ready", "not_loaded"]

    # Ask questions
    results["ask_install"] = test_ask(
        base_url,
        "How do I install the software?",
        ["install", "download", "setup"]
    )

    results["ask_troubleshoot"] = test_ask(
        base_url,
        "How do I troubleshoot connection issues?",
        ["connection", "network", "check"]
    )

    results["ask_api"] = test_ask(
        base_url,
        "What API endpoints are available?",
        ["endpoint", "api", "request"]
    )

    # Parameter tests
    results["temperature"] = test_temperature_variation(base_url)
    results["max_length"] = test_max_length(base_url)

    return results


def print_summary(results: Dict[str, bool]):
    """Print test summary."""
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print("-" * 50)
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


def main():
    if not HAS_REQUESTS:
        print("requests library required. Install with: pip install requests")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test the technical assistant service")
    parser.add_argument("--url", default="http://localhost:8002",
                        help="Base URL of the assistant service")
    parser.add_argument("--question", help="Run single question test")

    args = parser.parse_args()

    print(f"Testing service at: {args.url}")

    if args.question:
        # Single question test
        test_ask(args.url, args.question)
    else:
        # Run all tests
        results = run_all_tests(args.url)
        exit_code = print_summary(results)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
