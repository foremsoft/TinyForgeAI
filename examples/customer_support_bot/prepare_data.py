#!/usr/bin/env python3
"""
Data preparation script for Customer Support Bot.

This script handles data ingestion from multiple sources and converts
documents into Q&A pair format suitable for training.

Usage:
    python prepare_data.py --source sample --output data/support_faq_dataset/faq_data.jsonl
    python prepare_data.py --source confluence --space SUPPORT --output data/training.jsonl
    python prepare_data.py --source zendesk --file tickets.json --output data/training.jsonl
    python prepare_data.py --source markdown --dir ./docs --output data/training.jsonl
"""

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class Document:
    """Represents a document with content and metadata."""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None):
        self.content = content
        self.metadata = metadata or {}


class QAPair:
    """Represents a question-answer pair for training."""

    def __init__(
        self,
        question: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.question = question
        self.answer = answer
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to training format."""
        result = {"input": self.question, "output": self.answer}
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def fetch_documents(self) -> list[Document]:
        """Fetch documents from the data source."""
        pass


class SampleDataSource(DataSource):
    """Generate sample FAQ data for demonstration."""

    def fetch_documents(self) -> list[Document]:
        """Return sample support FAQ documents."""
        samples = [
            {
                "q": "How do I reset my password?",
                "a": "To reset your password: 1) Click 'Forgot Password' on the login page, 2) Enter your email address, 3) Check your inbox for the reset link (check spam folder if not found), 4) Click the link and create a new password. The link expires in 24 hours. If you don't receive the email within 5 minutes, try again or contact support.",
                "category": "account",
            },
            {
                "q": "What are your business hours?",
                "a": "Our customer support team is available Monday through Friday, 9 AM to 6 PM EST. For urgent issues outside these hours, please use our emergency support line at 1-800-SUPPORT or submit a ticket through our portal which will be addressed first thing the next business day.",
                "category": "general",
            },
            {
                "q": "How do I cancel my subscription?",
                "a": "To cancel your subscription: 1) Log into your account, 2) Go to Settings > Subscription, 3) Click 'Cancel Subscription', 4) Follow the prompts to confirm. Your access will continue until the end of your current billing period. You can reactivate anytime. Note: Annual subscriptions may be eligible for a prorated refund - contact billing@company.com.",
                "category": "billing",
            },
            {
                "q": "I was charged twice for my subscription",
                "a": "I apologize for the billing error. To resolve duplicate charges: 1) Please provide your account email and the last 4 digits of the card charged, 2) We will verify the duplicate transaction, 3) A refund will be initiated within 24 hours, 4) The refund will appear in your account within 3-5 business days depending on your bank. For immediate assistance, call our billing team at 1-800-BILLING.",
                "category": "billing",
            },
            {
                "q": "How do I upgrade my plan?",
                "a": "To upgrade your plan: 1) Log into your account, 2) Navigate to Settings > Subscription, 3) Click 'Change Plan', 4) Select your new plan, 5) Review the prorated charges and confirm. The upgrade takes effect immediately. You'll be charged the prorated difference for the remainder of your billing cycle. Enterprise plans require contacting sales@company.com.",
                "category": "billing",
            },
            {
                "q": "Where can I find my invoices?",
                "a": "Your invoices are available in your account: 1) Log into your account, 2) Go to Settings > Billing, 3) Click 'Invoice History', 4) Download individual invoices as PDF or export all invoices as CSV. Invoices are also emailed to your registered email address after each payment. For custom invoice requirements, contact billing@company.com.",
                "category": "billing",
            },
            {
                "q": "How do I add team members to my account?",
                "a": "To add team members: 1) Log into your admin account, 2) Go to Settings > Team, 3) Click 'Invite Member', 4) Enter their email and select their role (Admin, Editor, or Viewer), 5) Click 'Send Invite'. They'll receive an email to set up their account. Note: The number of team members depends on your plan. Check Settings > Subscription for your limit.",
                "category": "account",
            },
            {
                "q": "What payment methods do you accept?",
                "a": "We accept the following payment methods: 1) Credit/Debit Cards: Visa, MasterCard, American Express, Discover, 2) PayPal, 3) Bank Transfer (for annual Enterprise plans), 4) Wire Transfer (for Enterprise plans over $10,000). All payments are processed securely through Stripe. For invoicing options on Enterprise plans, contact sales@company.com.",
                "category": "billing",
            },
            {
                "q": "How do I export my data?",
                "a": "To export your data: 1) Log into your account, 2) Go to Settings > Data, 3) Click 'Export Data', 4) Select the data types you want to export (all data, specific projects, etc.), 5) Choose format (JSON, CSV, or ZIP archive), 6) Click 'Generate Export'. You'll receive an email with a download link within 24 hours for large exports, or immediate download for smaller datasets.",
                "category": "data",
            },
            {
                "q": "Is my data secure?",
                "a": "Yes, we take data security seriously. Our security measures include: 1) 256-bit SSL encryption for all data in transit, 2) AES-256 encryption for data at rest, 3) SOC 2 Type II certification, 4) GDPR compliance, 5) Regular third-party security audits, 6) Two-factor authentication available, 7) Role-based access controls. For our full security documentation, visit company.com/security.",
                "category": "security",
            },
            {
                "q": "How do I enable two-factor authentication?",
                "a": "To enable two-factor authentication (2FA): 1) Log into your account, 2) Go to Settings > Security, 3) Click 'Enable 2FA', 4) Choose your method: Authenticator App (recommended) or SMS, 5) Follow the setup prompts, 6) Save your backup codes in a secure location. Once enabled, you'll need to enter a code from your authenticator app or SMS when logging in.",
                "category": "security",
            },
            {
                "q": "My account is locked, what should I do?",
                "a": "If your account is locked: 1) Wait 30 minutes and try again (accounts auto-unlock after too many failed attempts), 2) Try resetting your password using 'Forgot Password', 3) Clear your browser cache and cookies, 4) Try a different browser or device. If still locked after these steps, contact support with your account email and we'll verify your identity and unlock your account within 1 business hour.",
                "category": "account",
            },
            {
                "q": "How do I contact customer support?",
                "a": "You can reach our support team through multiple channels: 1) Email: support@company.com (response within 24 hours), 2) Live Chat: Available on our website during business hours, 3) Phone: 1-800-SUPPORT (Mon-Fri, 9 AM - 6 PM EST), 4) Help Center: help.company.com for self-service articles, 5) Community Forum: community.company.com for peer support. Premium plans include priority support with 4-hour response time.",
                "category": "general",
            },
            {
                "q": "Do you offer refunds?",
                "a": "Our refund policy: 1) Monthly subscriptions: Full refund within 7 days of initial purchase, 2) Annual subscriptions: Prorated refund within 30 days, 3) Enterprise contracts: As specified in your agreement. To request a refund: Email billing@company.com with your account email and reason for refund. Refunds are processed within 5-7 business days. Note: Refunds are not available after the specified periods.",
                "category": "billing",
            },
            {
                "q": "How do I change my email address?",
                "a": "To change your email address: 1) Log into your account, 2) Go to Settings > Profile, 3) Click 'Edit' next to your email, 4) Enter your new email address, 5) Enter your password to confirm, 6) Click 'Save'. A verification email will be sent to your new address. Click the verification link to complete the change. Your old email will receive a notification of the change for security.",
                "category": "account",
            },
            {
                "q": "Why is the application running slowly?",
                "a": "If you're experiencing slow performance: 1) Check your internet connection speed at speedtest.net, 2) Clear your browser cache and cookies, 3) Disable browser extensions temporarily, 4) Try a different browser (we recommend Chrome or Firefox), 5) Check our status page at status.company.com for any ongoing issues. If problems persist, contact support with your browser version and a description of the slow actions.",
                "category": "technical",
            },
            {
                "q": "How do I integrate with other tools?",
                "a": "We offer multiple integration options: 1) Native Integrations: Slack, Microsoft Teams, Jira, GitHub, Salesforce - configure in Settings > Integrations, 2) Zapier: Connect with 3000+ apps via our Zapier integration, 3) REST API: Full API access for custom integrations - see docs.company.com/api, 4) Webhooks: Real-time event notifications - configure in Settings > Webhooks. Enterprise plans include custom integration development.",
                "category": "technical",
            },
            {
                "q": "What browsers are supported?",
                "a": "We support the following browsers (latest 2 versions): 1) Google Chrome (recommended), 2) Mozilla Firefox, 3) Microsoft Edge, 4) Safari (macOS/iOS), 5) Opera. For the best experience, we recommend Chrome with JavaScript enabled. Internet Explorer is not supported. Mobile browsers are supported on iOS Safari and Android Chrome.",
                "category": "technical",
            },
            {
                "q": "How do I delete my account?",
                "a": "To delete your account: 1) Export any data you want to keep first (Settings > Data > Export), 2) Cancel your subscription if active, 3) Go to Settings > Account, 4) Click 'Delete Account', 5) Enter your password and type 'DELETE' to confirm. Account deletion is permanent and cannot be undone. All data will be removed within 30 days per our data retention policy.",
                "category": "account",
            },
            {
                "q": "Do you have a mobile app?",
                "a": "Yes! Our mobile app is available for both iOS and Android: 1) iOS: Download from the App Store (search 'Company App'), 2) Android: Download from Google Play Store. The mobile app supports: viewing and editing content, notifications, offline access to recent items, and camera/photo upload. Some advanced features are only available on the web version.",
                "category": "general",
            },
        ]

        documents = []
        for sample in samples:
            doc = Document(
                content=json.dumps({"question": sample["q"], "answer": sample["a"]}),
                metadata={"category": sample["category"], "source": "sample"},
            )
            documents.append(doc)

        return documents


class ConfluenceDataSource(DataSource):
    """Fetch FAQ documents from Confluence."""

    def __init__(self, space: str, url: str | None = None, token: str | None = None):
        self.space = space
        self.url = url
        self.token = token

    def fetch_documents(self) -> list[Document]:
        """Fetch documents from Confluence space."""
        try:
            from connectors.confluence_connector import ConfluenceConnector

            connector = ConfluenceConnector(
                base_url=self.url or "",
                space_key=self.space,
                api_token=self.token or "",
            )
            pages = connector.fetch_pages()
            return [
                Document(content=page.content, metadata={"source": "confluence", "title": page.title})
                for page in pages
            ]
        except ImportError:
            print("Warning: Confluence connector not available. Using sample data.")
            return SampleDataSource().fetch_documents()
        except Exception as e:
            print(f"Warning: Failed to fetch from Confluence: {e}. Using sample data.")
            return SampleDataSource().fetch_documents()


class ZendeskDataSource(DataSource):
    """Fetch support tickets from Zendesk export."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def fetch_documents(self) -> list[Document]:
        """Parse Zendesk ticket export."""
        if not self.file_path.exists():
            print(f"Warning: Zendesk file not found: {self.file_path}. Using sample data.")
            return SampleDataSource().fetch_documents()

        documents = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tickets = data.get("tickets", data) if isinstance(data, dict) else data
        for ticket in tickets:
            if isinstance(ticket, dict):
                question = ticket.get("subject", ticket.get("question", ""))
                answer = ticket.get("description", ticket.get("answer", ""))
                if question and answer:
                    doc = Document(
                        content=json.dumps({"question": question, "answer": answer}),
                        metadata={"source": "zendesk", "ticket_id": ticket.get("id")},
                    )
                    documents.append(doc)

        return documents if documents else SampleDataSource().fetch_documents()


class MarkdownDataSource(DataSource):
    """Extract Q&A pairs from markdown documentation."""

    def __init__(self, directory: str):
        self.directory = Path(directory)

    def fetch_documents(self) -> list[Document]:
        """Parse markdown files for FAQ content."""
        if not self.directory.exists():
            print(f"Warning: Directory not found: {self.directory}. Using sample data.")
            return SampleDataSource().fetch_documents()

        documents = []
        for md_file in self.directory.glob("**/*.md"):
            content = md_file.read_text(encoding="utf-8")
            qa_pairs = self._extract_qa_pairs(content)
            for q, a in qa_pairs:
                doc = Document(
                    content=json.dumps({"question": q, "answer": a}),
                    metadata={"source": "markdown", "file": str(md_file.name)},
                )
                documents.append(doc)

        return documents if documents else SampleDataSource().fetch_documents()

    def _extract_qa_pairs(self, content: str) -> list[tuple[str, str]]:
        """Extract Q&A pairs from markdown content."""
        pairs = []

        # Pattern 1: Q: ... A: ... format
        qa_pattern = r"(?:Q:|Question:)\s*(.+?)\s*(?:A:|Answer:)\s*(.+?)(?=(?:Q:|Question:)|$)"
        matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
        for q, a in matches:
            pairs.append((q.strip(), a.strip()))

        # Pattern 2: ## Header followed by content (FAQ style)
        header_pattern = r"##\s*(.+?)\n\n(.+?)(?=\n##|\n#|$)"
        matches = re.findall(header_pattern, content, re.DOTALL)
        for header, body in matches:
            if "?" in header or any(
                kw in header.lower() for kw in ["how", "what", "why", "when", "where", "can", "do"]
            ):
                pairs.append((header.strip(), body.strip()))

        return pairs


class DocumentProcessor:
    """Process documents into Q&A pairs."""

    def process(self, documents: list[Document]) -> list[QAPair]:
        """Convert documents to Q&A pairs."""
        qa_pairs = []

        for doc in documents:
            try:
                data = json.loads(doc.content)
                question = data.get("question", data.get("q", ""))
                answer = data.get("answer", data.get("a", ""))
                if question and answer:
                    qa_pair = QAPair(
                        question=question,
                        answer=answer,
                        metadata=doc.metadata,
                    )
                    qa_pairs.append(qa_pair)
            except json.JSONDecodeError:
                # Try to extract Q&A from plain text
                pass

        return qa_pairs


class DatasetBuilder:
    """Build training dataset from Q&A pairs."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

    def build(self, qa_pairs: list[QAPair]) -> None:
        """Write Q&A pairs to JSONL file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")

        print(f"Dataset saved to {self.output_path}")
        print(f"Total Q&A pairs: {len(qa_pairs)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare training data for Customer Support Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_data.py --source sample --output data/support_faq_dataset/faq_data.jsonl
    python prepare_data.py --source confluence --space SUPPORT --output data/training.jsonl
    python prepare_data.py --source zendesk --file tickets.json --output data/training.jsonl
    python prepare_data.py --source markdown --dir ./docs --output data/training.jsonl
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["sample", "confluence", "zendesk", "markdown"],
        help="Data source type",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--space",
        type=str,
        help="Confluence space key (for confluence source)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Input file path (for zendesk source)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Input directory (for markdown source)",
    )
    parser.add_argument(
        "--confluence-url",
        type=str,
        help="Confluence base URL",
    )
    parser.add_argument(
        "--confluence-token",
        type=str,
        help="Confluence API token",
    )

    args = parser.parse_args()

    # Select data source
    if args.source == "sample":
        data_source = SampleDataSource()
    elif args.source == "confluence":
        if not args.space:
            parser.error("--space is required for confluence source")
        data_source = ConfluenceDataSource(
            space=args.space,
            url=args.confluence_url,
            token=args.confluence_token,
        )
    elif args.source == "zendesk":
        if not args.file:
            parser.error("--file is required for zendesk source")
        data_source = ZendeskDataSource(file_path=args.file)
    elif args.source == "markdown":
        if not args.dir:
            parser.error("--dir is required for markdown source")
        data_source = MarkdownDataSource(directory=args.dir)
    else:
        parser.error(f"Unknown source: {args.source}")

    # Process data
    print(f"Fetching documents from {args.source}...")
    documents = data_source.fetch_documents()
    print(f"Fetched {len(documents)} documents")

    print("Processing documents into Q&A pairs...")
    processor = DocumentProcessor()
    qa_pairs = processor.process(documents)
    print(f"Generated {len(qa_pairs)} Q&A pairs")

    print(f"Building dataset at {args.output}...")
    builder = DatasetBuilder(args.output)
    builder.build(qa_pairs)

    print("Done!")


if __name__ == "__main__":
    main()
