"""
Data Augmentation Strategies.

Various strategies for generating synthetic training data from examples.
"""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentedSample:
    """Represents an augmented training sample."""
    input: str
    output: str
    original_input: str
    original_output: str
    strategy: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class AugmentationStrategy(ABC):
    """Base class for data augmentation strategies."""

    name: str = "base"

    @abstractmethod
    def augment(
        self,
        input_text: str,
        output_text: str,
        n_variations: int = 5
    ) -> List[AugmentedSample]:
        """
        Generate augmented samples from an input-output pair.

        Args:
            input_text: Original input text.
            output_text: Original output text.
            n_variations: Number of variations to generate.

        Returns:
            List of augmented samples.
        """
        pass


class SynonymStrategy(AugmentationStrategy):
    """
    Augment data by replacing words with synonyms.

    Uses a built-in synonym dictionary for common words.
    No external dependencies required.
    """

    name = "synonym"

    # Common synonym mappings
    SYNONYMS = {
        # Question words
        "what": ["which", "what exactly"],
        "how": ["in what way", "by what means"],
        "why": ["for what reason", "what causes"],
        "when": ["at what time", "what time"],
        "where": ["in what place", "at what location"],
        "who": ["which person", "what person"],

        # Common verbs
        "help": ["assist", "support", "aid"],
        "get": ["obtain", "acquire", "receive"],
        "make": ["create", "build", "construct"],
        "use": ["utilize", "employ", "apply"],
        "find": ["locate", "discover", "identify"],
        "give": ["provide", "offer", "supply"],
        "tell": ["inform", "explain", "describe"],
        "show": ["display", "demonstrate", "present"],
        "know": ["understand", "comprehend", "realize"],
        "want": ["need", "require", "desire"],
        "see": ["view", "observe", "notice"],
        "come": ["arrive", "approach", "reach"],
        "go": ["proceed", "move", "travel"],
        "take": ["grab", "acquire", "obtain"],
        "put": ["place", "set", "position"],

        # Common adjectives
        "good": ["great", "excellent", "fine"],
        "bad": ["poor", "terrible", "awful"],
        "big": ["large", "huge", "enormous"],
        "small": ["tiny", "little", "compact"],
        "new": ["recent", "fresh", "latest"],
        "old": ["previous", "former", "past"],
        "easy": ["simple", "straightforward", "effortless"],
        "hard": ["difficult", "challenging", "tough"],
        "fast": ["quick", "rapid", "speedy"],
        "slow": ["gradual", "unhurried", "leisurely"],

        # Common nouns
        "problem": ["issue", "challenge", "difficulty"],
        "question": ["query", "inquiry", "request"],
        "answer": ["response", "reply", "solution"],
        "information": ["details", "data", "info"],
        "way": ["method", "approach", "manner"],
        "thing": ["item", "object", "element"],
        "time": ["moment", "period", "duration"],
        "person": ["individual", "user", "someone"],
        "place": ["location", "area", "spot"],
        "work": ["job", "task", "project"],

        # Business terms
        "customer": ["client", "user", "buyer"],
        "product": ["item", "service", "offering"],
        "price": ["cost", "rate", "fee"],
        "order": ["purchase", "request", "booking"],
        "shipping": ["delivery", "dispatch", "transport"],
        "return": ["refund", "exchange", "send back"],
        "account": ["profile", "membership", "registration"],
        "password": ["passcode", "credentials", "login"],
        "support": ["help", "assistance", "service"],
        "policy": ["rules", "terms", "guidelines"],
    }

    def __init__(self, replacement_prob: float = 0.3):
        """
        Initialize synonym strategy.

        Args:
            replacement_prob: Probability of replacing each eligible word.
        """
        self.replacement_prob = replacement_prob

    def augment(
        self,
        input_text: str,
        output_text: str,
        n_variations: int = 5
    ) -> List[AugmentedSample]:
        """Generate variations by replacing words with synonyms."""
        results = []

        for _ in range(n_variations):
            words = input_text.lower().split()
            new_words = []
            replacements_made = 0

            for word in words:
                # Clean word of punctuation for lookup
                clean_word = re.sub(r'[^\w]', '', word)

                if clean_word in self.SYNONYMS and random.random() < self.replacement_prob:
                    synonyms = self.SYNONYMS[clean_word]
                    replacement = random.choice(synonyms)
                    # Preserve original punctuation
                    new_word = word.replace(clean_word, replacement)
                    new_words.append(new_word)
                    replacements_made += 1
                else:
                    new_words.append(word)

            if replacements_made > 0:
                new_input = ' '.join(new_words)
                # Capitalize first letter
                new_input = new_input[0].upper() + new_input[1:] if new_input else new_input

                results.append(AugmentedSample(
                    input=new_input,
                    output=output_text,  # Keep output unchanged
                    original_input=input_text,
                    original_output=output_text,
                    strategy=self.name,
                    confidence=0.9,
                    metadata={"replacements": replacements_made}
                ))

        return results


class TemplateStrategy(AugmentationStrategy):
    """
    Augment data using template-based variations.

    Applies common question/statement templates to generate variations.
    """

    name = "template"

    # Question templates for different intents
    QUESTION_TEMPLATES = {
        "how": [
            "How do I {action}?",
            "How can I {action}?",
            "What's the best way to {action}?",
            "Can you explain how to {action}?",
            "I need help with {action}",
            "Help me {action}",
            "What's the process for {action}?",
            "Steps to {action}?",
        ],
        "what": [
            "What is {topic}?",
            "What does {topic} mean?",
            "Can you explain {topic}?",
            "Tell me about {topic}",
            "I want to know about {topic}",
            "Explain {topic} please",
            "What's {topic}?",
            "Define {topic}",
        ],
        "where": [
            "Where can I find {item}?",
            "Where is {item} located?",
            "How do I find {item}?",
            "I'm looking for {item}",
            "Can you tell me where {item} is?",
            "Location of {item}?",
        ],
        "when": [
            "When is {event}?",
            "What time is {event}?",
            "When does {event} happen?",
            "What's the schedule for {event}?",
            "Timing for {event}?",
        ],
        "why": [
            "Why is {reason}?",
            "What's the reason for {reason}?",
            "Can you explain why {reason}?",
            "I don't understand why {reason}",
        ],
        "can": [
            "Can I {action}?",
            "Is it possible to {action}?",
            "Am I able to {action}?",
            "Do you allow {action}?",
            "Is {action} allowed?",
        ],
        "do": [
            "Do you {topic}?",
            "Does your company {topic}?",
            "Is {topic} available?",
            "Do you offer {topic}?",
        ],
    }

    def __init__(self):
        """Initialize template strategy."""
        pass

    def _extract_intent_and_subject(self, text: str) -> Tuple[str, str]:
        """Extract the intent and subject from a question."""
        text_lower = text.lower().strip()

        # Try to match question patterns
        patterns = [
            (r"^how (?:do i|can i|to) (.+?)\??$", "how"),
            (r"^what (?:is|are|does) (.+?)\??$", "what"),
            (r"^where (?:can i find|is) (.+?)\??$", "where"),
            (r"^when (?:is|does|will) (.+?)\??$", "when"),
            (r"^why (?:is|does|do) (.+?)\??$", "why"),
            (r"^can i (.+?)\??$", "can"),
            (r"^do you (.+?)\??$", "do"),
        ]

        for pattern, intent in patterns:
            match = re.match(pattern, text_lower)
            if match:
                return intent, match.group(1)

        # Default: treat as general topic
        return "what", text_lower.rstrip("?")

    def augment(
        self,
        input_text: str,
        output_text: str,
        n_variations: int = 5
    ) -> List[AugmentedSample]:
        """Generate variations using templates."""
        results = []

        intent, subject = self._extract_intent_and_subject(input_text)
        templates = self.QUESTION_TEMPLATES.get(intent, self.QUESTION_TEMPLATES["what"])

        # Select random templates
        selected_templates = random.sample(
            templates,
            min(n_variations, len(templates))
        )

        placeholder = "{action}" if intent in ["how", "can"] else "{topic}" if intent in ["what", "do"] else "{item}" if intent == "where" else "{event}" if intent == "when" else "{reason}"

        for template in selected_templates:
            try:
                new_input = template.replace(placeholder, subject)
                # Also try other placeholders
                for ph in ["{action}", "{topic}", "{item}", "{event}", "{reason}"]:
                    new_input = new_input.replace(ph, subject)

                if new_input.lower() != input_text.lower():
                    results.append(AugmentedSample(
                        input=new_input,
                        output=output_text,
                        original_input=input_text,
                        original_output=output_text,
                        strategy=self.name,
                        confidence=0.85,
                        metadata={"template": template, "intent": intent}
                    ))
            except Exception as e:
                logger.warning(f"Template augmentation failed: {e}")
                continue

        return results


class ParaphraseStrategy(AugmentationStrategy):
    """
    Augment data using rule-based paraphrasing.

    Applies common paraphrase patterns without requiring external models.
    """

    name = "paraphrase"

    # Paraphrase rules: (pattern, replacement)
    PARAPHRASE_RULES = [
        # Contractions
        (r"\bcan't\b", "cannot"),
        (r"\bwon't\b", "will not"),
        (r"\bdon't\b", "do not"),
        (r"\bdoesn't\b", "does not"),
        (r"\bisn't\b", "is not"),
        (r"\baren't\b", "are not"),
        (r"\bwasn't\b", "was not"),
        (r"\bweren't\b", "were not"),
        (r"\bhaven't\b", "have not"),
        (r"\bhasn't\b", "has not"),
        (r"\bhadn't\b", "had not"),
        (r"\bwouldn't\b", "would not"),
        (r"\bcouldn't\b", "could not"),
        (r"\bshouldn't\b", "should not"),
        (r"\bi'm\b", "I am"),
        (r"\byou're\b", "you are"),
        (r"\bwe're\b", "we are"),
        (r"\bthey're\b", "they are"),
        (r"\bit's\b", "it is"),
        (r"\bthat's\b", "that is"),
        (r"\bwhat's\b", "what is"),
        (r"\bwhere's\b", "where is"),
        (r"\bwho's\b", "who is"),
        (r"\bhow's\b", "how is"),
        (r"\bi've\b", "I have"),
        (r"\byou've\b", "you have"),
        (r"\bwe've\b", "we have"),
        (r"\bthey've\b", "they have"),
        (r"\bi'd\b", "I would"),
        (r"\byou'd\b", "you would"),
        (r"\bwe'd\b", "we would"),
        (r"\bthey'd\b", "they would"),
        (r"\bi'll\b", "I will"),
        (r"\byou'll\b", "you will"),
        (r"\bwe'll\b", "we will"),
        (r"\bthey'll\b", "they will"),

        # Expand contractions (reverse)
        (r"\bcannot\b", "can't"),
        (r"\bwill not\b", "won't"),
        (r"\bdo not\b", "don't"),
        (r"\bdoes not\b", "doesn't"),

        # Common phrase variations
        (r"\bplease\b", "kindly"),
        (r"\bkindly\b", "please"),
        (r"\bthanks\b", "thank you"),
        (r"\bthank you\b", "thanks"),
        (r"\bhi\b", "hello"),
        (r"\bhello\b", "hi"),
        (r"\bhey\b", "hi"),

        # Question reformulations
        (r"^can you tell me", "I'd like to know"),
        (r"^I'd like to know", "can you tell me"),
        (r"^I want to", "I'd like to"),
        (r"^I'd like to", "I want to"),
        (r"^could you", "can you"),
        (r"^can you", "could you"),
        (r"^would you", "can you"),
    ]

    def augment(
        self,
        input_text: str,
        output_text: str,
        n_variations: int = 5
    ) -> List[AugmentedSample]:
        """Generate paraphrased variations."""
        results = []
        seen = {input_text.lower()}

        # Try each rule and collect unique variations
        for pattern, replacement in self.PARAPHRASE_RULES:
            new_input = re.sub(pattern, replacement, input_text, flags=re.IGNORECASE)

            if new_input.lower() not in seen:
                seen.add(new_input.lower())
                results.append(AugmentedSample(
                    input=new_input,
                    output=output_text,
                    original_input=input_text,
                    original_output=output_text,
                    strategy=self.name,
                    confidence=0.95,
                    metadata={"rule": f"{pattern} -> {replacement}"}
                ))

                if len(results) >= n_variations:
                    break

        return results[:n_variations]


class BackTranslationStrategy(AugmentationStrategy):
    """
    Simulate back-translation augmentation.

    Uses rule-based transformations to simulate the effect of
    translating to another language and back. This creates
    natural-sounding variations without requiring translation APIs.
    """

    name = "back_translation"

    # Simulated translation artifacts
    TRANSFORMATIONS = [
        # Word order changes (common in translation)
        (r"(\w+) (\w+ly)", r"\2 \1"),  # Adverb movement

        # Article variations (common translation artifact)
        (r"\bthe\b", "a"),
        (r"\ba\b", "the"),
        (r"\ban\b", "the"),

        # Formality shifts
        (r"\byou\b", "one"),
        (r"\bone\b", "you"),

        # Passive/active voice hints
        (r"^I ", ""),
        (r"^", "I "),
    ]

    def augment(
        self,
        input_text: str,
        output_text: str,
        n_variations: int = 5
    ) -> List[AugmentedSample]:
        """Generate back-translation style variations."""
        results = []
        seen = {input_text.lower()}

        for _ in range(n_variations * 2):  # Try more to get enough unique ones
            # Apply random subset of transformations
            new_input = input_text
            transformations_applied = []

            for pattern, replacement in random.sample(
                self.TRANSFORMATIONS,
                min(2, len(self.TRANSFORMATIONS))
            ):
                if random.random() < 0.5:
                    new_text = re.sub(pattern, replacement, new_input, count=1)
                    if new_text != new_input:
                        new_input = new_text
                        transformations_applied.append(f"{pattern}")

            # Clean up and validate
            new_input = ' '.join(new_input.split())  # Normalize whitespace
            if new_input and new_input[0].islower():
                new_input = new_input[0].upper() + new_input[1:]

            if new_input.lower() not in seen and len(new_input) > 3:
                seen.add(new_input.lower())
                results.append(AugmentedSample(
                    input=new_input,
                    output=output_text,
                    original_input=input_text,
                    original_output=output_text,
                    strategy=self.name,
                    confidence=0.7,
                    metadata={"transformations": transformations_applied}
                ))

            if len(results) >= n_variations:
                break

        return results


class LLMStrategy(AugmentationStrategy):
    """
    Augment data using LLM APIs (Claude, OpenAI, etc.).

    This strategy calls an external LLM to generate high-quality
    variations of training data. Requires API credentials.
    """

    name = "llm"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM strategy.

        Args:
            provider: LLM provider ("anthropic" or "openai").
            model: Model name to use.
            api_key: API key (or set via environment variable).
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client

        import os

        if self.provider == "anthropic":
            try:
                import anthropic
                api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")

        elif self.provider == "openai":
            try:
                import openai
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def augment(
        self,
        input_text: str,
        output_text: str,
        n_variations: int = 5
    ) -> List[AugmentedSample]:
        """Generate variations using LLM."""
        client = self._get_client()

        prompt = f"""Generate {n_variations} different ways to ask the same question or make the same request.

Original: "{input_text}"

Requirements:
- Keep the same meaning/intent
- Vary the phrasing, formality, and structure
- Make them sound natural
- Output as JSON array of strings

Output format:
["variation 1", "variation 2", ...]"""

        try:
            if self.provider == "anthropic":
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:  # openai
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                content = response.choices[0].message.content

            # Parse JSON response
            # Find JSON array in response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                variations = json.loads(match.group())
            else:
                logger.warning(f"Could not parse LLM response: {content}")
                return []

            results = []
            for var in variations[:n_variations]:
                if isinstance(var, str) and var.strip():
                    results.append(AugmentedSample(
                        input=var.strip(),
                        output=output_text,
                        original_input=input_text,
                        original_output=output_text,
                        strategy=self.name,
                        confidence=0.95,
                        metadata={"model": self.model, "provider": self.provider}
                    ))

            return results

        except Exception as e:
            logger.error(f"LLM augmentation failed: {e}")
            return []
