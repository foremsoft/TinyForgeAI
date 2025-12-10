#!/usr/bin/env python3
"""
Test Script for Internal Knowledge Search

Tests the search service endpoints and validates functionality.

Usage:
    python test_search.py --url http://localhost:8001
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
            print("  ✓ Health check passed")
            return True
        else:
            print(f"  ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False


def test_status(base_url: str) -> Dict[str, Any]:
    """Test status endpoint."""
    print("\n[TEST] Index Status")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        data = response.json()

        print(f"  Status: {data.get('status')}")
        print(f"  Documents: {data.get('num_documents')}")
        print(f"  Model: {data.get('embedding_model')}")

        if data.get("status") == "ready":
            print("  ✓ Status check passed")
        else:
            print("  ⚠ Index not ready")

        return data
    except Exception as e:
        print(f"  ✗ Status check failed: {e}")
        return {}


def test_search(base_url: str, query: str, expected_min_results: int = 1) -> bool:
    """Test search endpoint."""
    print(f"\n[TEST] Search: '{query}'")
    try:
        response = requests.post(
            f"{base_url}/search",
            json={"query": query, "top_k": 5},
            timeout=30
        )

        if response.status_code != 200:
            print(f"  ✗ Search failed: {response.status_code}")
            return False

        data = response.json()
        results = data.get("results", [])
        search_time = data.get("search_time_ms", 0)

        print(f"  Results: {len(results)}")
        print(f"  Search time: {search_time}ms")

        if results:
            print(f"  Top result score: {results[0].get('score', 0):.3f}")
            print(f"  Top result preview: {results[0].get('text', '')[:100]}...")

        if len(results) >= expected_min_results:
            print("  ✓ Search test passed")
            return True
        else:
            print(f"  ⚠ Expected at least {expected_min_results} results")
            return False

    except Exception as e:
        print(f"  ✗ Search failed: {e}")
        return False


def test_answer(base_url: str, question: str) -> bool:
    """Test answer generation endpoint."""
    print(f"\n[TEST] Answer: '{question}'")
    try:
        response = requests.post(
            f"{base_url}/answer",
            json={"question": question, "include_sources": True},
            timeout=60
        )

        if response.status_code != 200:
            print(f"  ✗ Answer generation failed: {response.status_code}")
            return False

        data = response.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        gen_time = data.get("generation_time_ms", 0)

        print(f"  Generation time: {gen_time}ms")
        print(f"  Sources used: {len(sources)}")
        print(f"  Answer preview: {answer[:200]}...")

        if answer:
            print("  ✓ Answer test passed")
            return True
        else:
            print("  ⚠ Empty answer generated")
            return False

    except Exception as e:
        print(f"  ✗ Answer generation failed: {e}")
        return False


def test_search_with_filters(base_url: str) -> bool:
    """Test search with metadata filters."""
    print("\n[TEST] Search with Filters")
    try:
        response = requests.post(
            f"{base_url}/search",
            json={
                "query": "policy",
                "top_k": 5,
                "filters": {"source": "local"}
            },
            timeout=30
        )

        if response.status_code != 200:
            print(f"  ✗ Filtered search failed: {response.status_code}")
            return False

        data = response.json()
        results = data.get("results", [])

        print(f"  Results with filter: {len(results)}")

        # Verify all results match filter
        all_match = all(
            r.get("metadata", {}).get("source") == "local"
            for r in results
        )

        if all_match:
            print("  ✓ Filter test passed")
            return True
        else:
            print("  ⚠ Some results don't match filter")
            return False

    except Exception as e:
        print(f"  ✗ Filtered search failed: {e}")
        return False


def run_all_tests(base_url: str) -> Dict[str, bool]:
    """Run all tests and return results."""
    results = {}

    # Basic connectivity
    results["health"] = test_health(base_url)

    if not results["health"]:
        print("\n✗ Service not reachable. Make sure the service is running.")
        return results

    # Status check
    status = test_status(base_url)
    results["status"] = status.get("status") == "ready"

    if not results["status"]:
        print("\n⚠ Index not ready. Run index_documents.py first.")
        return results

    # Search tests
    results["search_vacation"] = test_search(base_url, "vacation policy")
    results["search_password"] = test_search(base_url, "how to reset password")
    results["search_security"] = test_search(base_url, "data security encryption")

    # Answer generation tests
    results["answer_vacation"] = test_answer(base_url, "What is the vacation policy?")
    results["answer_support"] = test_answer(base_url, "How do I contact support?")

    # Filter test
    results["search_filtered"] = test_search_with_filters(base_url)

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

    parser = argparse.ArgumentParser(description="Test the knowledge search service")
    parser.add_argument("--url", default="http://localhost:8001",
                        help="Base URL of the search service")
    parser.add_argument("--query", help="Run single search query")
    parser.add_argument("--question", help="Run single answer generation")

    args = parser.parse_args()

    print(f"Testing service at: {args.url}")

    if args.query:
        # Single search test
        test_search(args.url, args.query)
    elif args.question:
        # Single answer test
        test_answer(args.url, args.question)
    else:
        # Run all tests
        results = run_all_tests(args.url)
        exit_code = print_summary(results)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
