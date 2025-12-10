#!/usr/bin/env python3
"""
Data Preparation Script for Technical Manual Assistant

This script generates Q&A training pairs from technical documentation.

Usage:
    python prepare_data.py --input ./data/sample_manuals --output ./data/training_data.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml


class ManualParser:
    """Parse technical manuals into sections."""

    def __init__(self):
        self.section_patterns = [
            r'^#{1,3}\s+(.+)$',           # Markdown headers
            r'^(.+)\n[=-]{3,}$',          # Underlined headers
            r'^([A-Z][A-Z\s]+)$',         # ALL CAPS headers
            r'^(\d+\.)\s+(.+)$',          # Numbered sections
        ]

    def parse_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse a file into sections."""
        ext = file_path.suffix.lower()

        if ext in ['.txt', '.md']:
            content = file_path.read_text(encoding='utf-8')
            return self._parse_text(content, str(file_path))
        elif ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)

        return []

    def _parse_text(self, content: str, source: str) -> List[Dict[str, str]]:
        """Parse text content into sections."""
        sections = []
        current_section = None
        current_content = []

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for underlined headers (need to look ahead)
            if i + 1 < len(lines) and re.match(r'^[=-]{3,}$', lines[i + 1]):
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip(),
                        'source': source
                    })
                current_section = line.strip()
                current_content = []
                i += 2
                continue

            # Check for other header patterns
            header_match = None
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    header_match = match
                    break

            if header_match:
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip(),
                        'source': source
                    })
                current_section = header_match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

            i += 1

        # Add last section
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content).strip(),
                'source': source
            })

        return sections

    def _parse_pdf(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse PDF file into sections."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or '')
            content = '\n\n'.join(text)
            return self._parse_text(content, str(file_path))
        except ImportError:
            print(f"PyPDF2 not installed, skipping {file_path}")
            return []

    def _parse_docx(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse DOCX file into sections."""
        try:
            from docx import Document
            doc = Document(str(file_path))
            content = '\n\n'.join([p.text for p in doc.paragraphs])
            return self._parse_text(content, str(file_path))
        except ImportError:
            print(f"python-docx not installed, skipping {file_path}")
            return []


class QAGenerator:
    """Generate Q&A pairs from document sections."""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.question_templates = config.get('question_templates', [
            "How do I {action}?",
            "What is the procedure for {action}?",
            "Can you explain {topic}?",
            "What are the steps to {action}?",
        ])
        self.min_section_length = config.get('min_section_length', 100)
        self.qa_per_section = config.get('qa_per_section', 3)

    def generate_qa_pairs(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Generate Q&A pairs from sections."""
        qa_pairs = []

        for section in sections:
            title = section['title']
            content = section['content']

            # Skip short sections
            if len(content) < self.min_section_length:
                continue

            # Generate questions based on section title
            questions = self._generate_questions(title, content)

            for question in questions[:self.qa_per_section]:
                qa_pairs.append({
                    'input': question,
                    'output': content,
                    'metadata': {
                        'section': title,
                        'source': section.get('source', 'unknown')
                    }
                })

        return qa_pairs

    def _generate_questions(self, title: str, content: str) -> List[str]:
        """Generate questions based on section title and content."""
        questions = []
        title_lower = title.lower()

        # Direct questions from title
        if 'install' in title_lower:
            questions.extend([
                f"How do I install the software?",
                f"What are the installation steps?",
                f"How do I set up the system?",
            ])
        elif 'troubleshoot' in title_lower or 'problem' in title_lower:
            questions.extend([
                f"How do I troubleshoot issues?",
                f"What should I do if something goes wrong?",
                f"How do I fix common problems?",
            ])
        elif 'config' in title_lower or 'setting' in title_lower:
            questions.extend([
                f"How do I configure the system?",
                f"What configuration options are available?",
                f"How do I change settings?",
            ])
        elif 'api' in title_lower or 'endpoint' in title_lower:
            questions.extend([
                f"What API endpoints are available?",
                f"How do I use the API?",
                f"What is the API format?",
            ])
        elif 'require' in title_lower:
            questions.extend([
                f"What are the system requirements?",
                f"What do I need to run the software?",
                f"What are the prerequisites?",
            ])
        elif 'error' in title_lower:
            questions.extend([
                f"What does this error mean?",
                f"How do I fix errors?",
                f"What causes errors?",
            ])
        else:
            # Generic questions
            action = title_lower.replace('_', ' ').replace('-', ' ')
            questions.extend([
                f"What is {action}?",
                f"How does {action} work?",
                f"Can you explain {action}?",
            ])

        # Add content-based questions
        if 'step' in content.lower() or re.search(r'\d+\.\s', content):
            questions.append(f"What are the steps for {title.lower()}?")

        if 'command' in content.lower() or '```' in content:
            questions.append(f"What commands do I need?")

        return questions


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'configs' / 'assistant_config.yaml'

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get('data_prep', {})

    return {}


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from technical manuals')
    parser.add_argument('--input', required=True, help='Input directory or file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--qa-per-section', type=int, help='Q&A pairs per section')
    parser.add_argument('--min-length', type=int, help='Minimum section length')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.qa_per_section:
        config['qa_per_section'] = args.qa_per_section
    if args.min_length:
        config['min_section_length'] = args.min_length

    # Parse input files
    input_path = Path(args.input)
    parser = ManualParser()
    all_sections = []

    print(f"\n=== Parsing documents from {input_path} ===\n")

    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob('*.txt')) + \
                list(input_path.rglob('*.md')) + \
                list(input_path.rglob('*.pdf')) + \
                list(input_path.rglob('*.docx'))

    for file_path in files:
        print(f"Parsing: {file_path.name}")
        sections = parser.parse_file(file_path)
        all_sections.extend(sections)
        print(f"  Found {len(sections)} sections")

    print(f"\nTotal sections: {len(all_sections)}")

    # Generate Q&A pairs
    print(f"\n=== Generating Q&A pairs ===\n")

    generator = QAGenerator(config)
    qa_pairs = generator.generate_qa_pairs(all_sections)

    print(f"Generated {len(qa_pairs)} Q&A pairs")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            # Write in JSONL format (without metadata for training)
            f.write(json.dumps({
                'input': qa['input'],
                'output': qa['output']
            }, ensure_ascii=False) + '\n')

    print(f"\n=== Data preparation complete! ===")
    print(f"Output: {output_path}")
    print(f"Total Q&A pairs: {len(qa_pairs)}")

    # Show sample
    if qa_pairs:
        print(f"\nSample Q&A pair:")
        print(f"  Q: {qa_pairs[0]['input']}")
        print(f"  A: {qa_pairs[0]['output'][:200]}...")


if __name__ == '__main__':
    main()
