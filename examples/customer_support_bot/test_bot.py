#!/usr/bin/env python3
"""
Testing script for Customer Support Bot.

This script validates model performance with test questions and
demonstrates how to interact with the deployed bot.

Usage:
    python test_bot.py --url http://localhost:8000
    python test_bot.py --model ./output/support_bot
    python test_bot.py --url http://localhost:8000 --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Test questions for the support bot
TEST_QUESTIONS = [
    {
        "question": "How do I reset my password?",
        "expected_keywords": ["password", "reset", "email", "link"],
        "category": "account",
    },
    {
        "question": "I was charged twice this month",
        "expected_keywords": ["refund", "billing", "charge", "apologize"],
        "category": "billing",
    },
    {
        "question": "What are your business hours?",
        "expected_keywords": ["hours", "monday", "friday", "support"],
        "category": "general",
    },
    {
        "question": "How do I cancel my subscription?",
        "expected_keywords": ["cancel", "subscription", "settings", "billing"],
        "category": "billing",
    },
    {
        "question": "I can't log into my account",
        "expected_keywords": ["password", "reset", "account", "login"],
        "category": "account",
    },
    {
        "question": "How do I upgrade my plan?",
        "expected_keywords": ["upgrade", "plan", "settings", "subscription"],
        "category": "billing",
    },
    {
        "question": "Is my data secure?",
        "expected_keywords": ["secure", "encryption", "privacy", "data"],
        "category": "security",
    },
    {
        "question": "How do I contact customer support?",
        "expected_keywords": ["support", "contact", "email", "phone"],
        "category": "general",
    },
]


class BotTester:
    """Test the customer support bot."""

    def __init__(self, url: str | None = None, model_path: str | None = None):
        self.url = url
        self.model_path = model_path
        self.model_loader = None

        if model_path and not url:
            from deploy import ModelLoader

            self.model_loader = ModelLoader(model_path)

    def predict(
        self,
        text: str,
        max_length: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Get a prediction from the bot."""
        if self.url:
            return self._predict_via_api(text, max_length, temperature)
        elif self.model_loader:
            return self.model_loader.predict(text, max_length, temperature)
        else:
            raise ValueError("Either url or model_path must be provided")

    def _predict_via_api(
        self,
        text: str,
        max_length: int,
        temperature: float,
    ) -> str:
        """Send prediction request to API server."""
        response = requests.post(
            f"{self.url}/predict",
            json={
                "input": text,
                "max_length": max_length,
                "temperature": temperature,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["output"]

    def check_health(self) -> dict[str, Any]:
        """Check API health status."""
        if not self.url:
            return {"status": "local", "model_loaded": self.model_loader is not None}

        response = requests.get(f"{self.url}/health", timeout=10)
        response.raise_for_status()
        return response.json()


def evaluate_response(response: str, expected_keywords: list[str]) -> dict[str, Any]:
    """Evaluate a response against expected keywords."""
    response_lower = response.lower()
    found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
    coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    return {
        "found_keywords": found_keywords,
        "missing_keywords": [kw for kw in expected_keywords if kw.lower() not in response_lower],
        "coverage": coverage,
        "response_length": len(response),
        "passed": coverage >= 0.5,  # Pass if at least 50% keywords found
    }


def run_tests(tester: BotTester, verbose: bool = True) -> dict[str, Any]:
    """Run all test questions and evaluate responses."""
    results = {
        "total": len(TEST_QUESTIONS),
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    print("=" * 60)
    print("Customer Support Bot - Test Suite")
    print("=" * 60)

    # Check health first
    try:
        health = tester.check_health()
        print(f"Health check: {health.get('status', 'unknown')}")
        print(f"Model loaded: {health.get('model_loaded', 'unknown')}")
    except Exception as e:
        print(f"Warning: Health check failed: {e}")

    print("-" * 60)

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        category = test["category"]

        if verbose:
            print(f"\nTest {i}/{len(TEST_QUESTIONS)} [{category}]")
            print(f"Q: {question}")

        try:
            response = tester.predict(question)
            evaluation = evaluate_response(response, expected_keywords)

            if verbose:
                print(f"A: {response[:200]}..." if len(response) > 200 else f"A: {response}")
                print(f"Keywords found: {evaluation['found_keywords']}")
                print(f"Coverage: {evaluation['coverage']:.0%}")
                print(f"Status: {'PASS' if evaluation['passed'] else 'FAIL'}")

            if evaluation["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

            results["details"].append(
                {
                    "question": question,
                    "category": category,
                    "response": response,
                    "evaluation": evaluation,
                }
            )

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results["failed"] += 1
            results["details"].append(
                {
                    "question": question,
                    "category": category,
                    "error": str(e),
                }
            )

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {results['passed'] / results['total']:.0%}")

    return results


def interactive_mode(tester: BotTester):
    """Run interactive chat mode."""
    print("=" * 60)
    print("Customer Support Bot - Interactive Mode")
    print("=" * 60)
    print("Type your questions and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            response = tester.predict(question)
            print(f"\nBot: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Customer Support Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test against running API server
    python test_bot.py --url http://localhost:8000

    # Test against local model
    python test_bot.py --model ./output/support_bot

    # Interactive mode
    python test_bot.py --url http://localhost:8000 --interactive

    # Run tests and save results
    python test_bot.py --url http://localhost:8000 --output results.json
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        help="URL of the running support bot API",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model directory (for local testing)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save test results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if not args.url and not args.model:
        parser.error("Either --url or --model is required")

    # Create tester
    tester = BotTester(url=args.url, model_path=args.model)

    if args.interactive:
        interactive_mode(tester)
    else:
        results = run_tests(tester, verbose=not args.quiet)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        # Exit with error code if tests failed
        sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
