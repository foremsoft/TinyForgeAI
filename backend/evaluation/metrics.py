"""
TinyForgeAI Evaluation Metrics

Provides standard NLP evaluation metrics for model assessment:
- BLEU: Machine translation quality
- ROUGE: Summarization quality
- Accuracy/F1: Classification tasks
- Perplexity: Language model quality
- Exact Match: QA tasks

All metrics work with optional dependencies and provide fallbacks.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Check for optional dependencies
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    logger.debug("NLTK not available, using simple BLEU implementation")

ROUGE_AVAILABLE = False
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    logger.debug("rouge-score not available, using simple ROUGE implementation")

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.debug("PyTorch not available, perplexity computation limited")


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with basic normalization."""
    text = text.lower().strip()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-gram counts from tokens."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU score for predictions vs references.

    Args:
        predictions: List of predicted strings
        references: List of reference strings (or list of multiple references)
        max_n: Maximum n-gram to consider (default 4 for BLEU-4)
        smoothing: Apply smoothing for short sentences

    Returns:
        Dictionary with BLEU scores (bleu, bleu_1, bleu_2, bleu_3, bleu_4)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # Normalize references to list of lists
    normalized_refs = []
    for ref in references:
        if isinstance(ref, str):
            normalized_refs.append([ref])
        else:
            normalized_refs.append(ref)

    if NLTK_AVAILABLE:
        # Use NLTK for more accurate BLEU
        smoothing_fn = SmoothingFunction().method1 if smoothing else None

        # Tokenize
        pred_tokens = [_tokenize(p) for p in predictions]
        ref_tokens = [[_tokenize(r) for r in refs] for refs in normalized_refs]

        # Compute individual BLEU scores
        bleu_scores = []
        for i in range(1, max_n + 1):
            weights = tuple([1.0/i] * i + [0.0] * (max_n - i))
            try:
                score = corpus_bleu(
                    ref_tokens, pred_tokens,
                    weights=weights,
                    smoothing_function=smoothing_fn
                )
            except (ZeroDivisionError, ValueError):
                score = 0.0
            bleu_scores.append(score)

        # Compute overall BLEU (BLEU-4)
        try:
            overall_bleu = corpus_bleu(
                ref_tokens, pred_tokens,
                smoothing_function=smoothing_fn
            )
        except (ZeroDivisionError, ValueError):
            overall_bleu = 0.0

        return {
            "bleu": overall_bleu,
            "bleu_1": bleu_scores[0] if len(bleu_scores) > 0 else 0.0,
            "bleu_2": bleu_scores[1] if len(bleu_scores) > 1 else 0.0,
            "bleu_3": bleu_scores[2] if len(bleu_scores) > 2 else 0.0,
            "bleu_4": bleu_scores[3] if len(bleu_scores) > 3 else 0.0,
        }
    else:
        # Simple implementation
        return _compute_bleu_simple(predictions, normalized_refs, max_n, smoothing)


def _compute_bleu_simple(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, float]:
    """Simple BLEU implementation without NLTK."""

    total_matches = [0] * max_n
    total_counts = [0] * max_n
    pred_lengths = 0
    ref_lengths = 0

    for pred, refs in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        pred_lengths += len(pred_tokens)

        # Find closest reference length
        ref_lens = [len(_tokenize(r)) for r in refs]
        closest_len = min(ref_lens, key=lambda x: abs(x - len(pred_tokens)))
        ref_lengths += closest_len

        for n in range(1, max_n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)

            # Get max counts from all references
            max_ref_counts: Counter = Counter()
            for ref in refs:
                ref_tokens = _tokenize(ref)
                ref_ngrams = _get_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            # Clip prediction counts
            clipped_counts = 0
            for ngram, count in pred_ngrams.items():
                clipped_counts += min(count, max_ref_counts.get(ngram, 0))

            total_matches[n-1] += clipped_counts
            total_counts[n-1] += sum(pred_ngrams.values())

    # Compute precisions with smoothing
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        elif total_matches[n] == 0:
            if smoothing:
                precisions.append(1.0 / (total_counts[n] + 1))
            else:
                precisions.append(0.0)
        else:
            precisions.append(total_matches[n] / total_counts[n])

    # Brevity penalty
    if pred_lengths == 0:
        bp = 0.0
    elif pred_lengths >= ref_lengths:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_lengths / pred_lengths)

    # Geometric mean of precisions
    bleu_scores = {}
    for n in range(1, max_n + 1):
        if all(p > 0 for p in precisions[:n]):
            log_sum = sum(math.log(p) for p in precisions[:n]) / n
            bleu_scores[f"bleu_{n}"] = bp * math.exp(log_sum)
        else:
            bleu_scores[f"bleu_{n}"] = 0.0

    bleu_scores["bleu"] = bleu_scores.get("bleu_4", 0.0)

    return bleu_scores


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE scores for summarization evaluation.

    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        rouge_types: ROUGE variants to compute (default: rouge1, rouge2, rougeL)

    Returns:
        Dictionary with ROUGE scores (precision, recall, fmeasure for each type)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    if len(predictions) == 0:
        return {rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for rt in rouge_types}

    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

        results = {rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for rt in rouge_types}

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for rt in rouge_types:
                results[rt]["precision"] += scores[rt].precision
                results[rt]["recall"] += scores[rt].recall
                results[rt]["fmeasure"] += scores[rt].fmeasure

        # Average
        n = len(predictions)
        for rt in rouge_types:
            for metric in ["precision", "recall", "fmeasure"]:
                results[rt][metric] /= n

        return results
    else:
        # Simple ROUGE-N implementation
        return _compute_rouge_simple(predictions, references, rouge_types)


def _compute_rouge_simple(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """Simple ROUGE implementation without rouge-score library."""

    results = {rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)

        for rt in rouge_types:
            if rt == "rouge1":
                n = 1
            elif rt == "rouge2":
                n = 2
            elif rt == "rougeL":
                # LCS-based ROUGE
                lcs_len = _lcs_length(pred_tokens, ref_tokens)
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    precision = lcs_len / len(pred_tokens)
                    recall = lcs_len / len(ref_tokens)
                    if precision + recall > 0:
                        fmeasure = 2 * precision * recall / (precision + recall)
                    else:
                        fmeasure = 0.0
                else:
                    precision = recall = fmeasure = 0.0
                results[rt]["precision"] += precision
                results[rt]["recall"] += recall
                results[rt]["fmeasure"] += fmeasure
                continue
            else:
                continue

            # N-gram based ROUGE
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            overlap = sum((pred_ngrams & ref_ngrams).values())
            pred_total = sum(pred_ngrams.values())
            ref_total = sum(ref_ngrams.values())

            if pred_total > 0:
                precision = overlap / pred_total
            else:
                precision = 0.0

            if ref_total > 0:
                recall = overlap / ref_total
            else:
                recall = 0.0

            if precision + recall > 0:
                fmeasure = 2 * precision * recall / (precision + recall)
            else:
                fmeasure = 0.0

            results[rt]["precision"] += precision
            results[rt]["recall"] += recall
            results[rt]["fmeasure"] += fmeasure

    # Average
    n = len(predictions)
    if n > 0:
        for rt in rouge_types:
            for metric in ["precision", "recall", "fmeasure"]:
                results[rt][metric] /= n

    return results


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]


def compute_accuracy(
    predictions: List[Any],
    references: List[Any],
) -> Dict[str, float]:
    """
    Compute accuracy for classification tasks.

    Args:
        predictions: List of predicted labels
        references: List of true labels

    Returns:
        Dictionary with accuracy score
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    total = len(predictions)

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


def compute_f1(
    predictions: List[Any],
    references: List[Any],
    average: str = "macro",
    labels: Optional[List[Any]] = None,
) -> Dict[str, float]:
    """
    Compute F1 score for classification tasks.

    Args:
        predictions: List of predicted labels
        references: List of true labels
        average: Averaging method ("micro", "macro", "weighted")
        labels: List of labels to consider

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Get unique labels
    if labels is None:
        labels = sorted(set(predictions) | set(references))

    # Compute per-class metrics
    class_metrics = {}
    for label in labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    # Aggregate based on average type
    if average == "micro":
        total_tp = sum(1 for p, r in zip(predictions, references) if p == r)
        total_fp = len(predictions) - total_tp
        total_fn = len(references) - total_tp

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "weighted":
        total_support = sum(m["support"] for m in class_metrics.values())
        if total_support > 0:
            precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support
            recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support
            f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support
        else:
            precision = recall = f1 = 0.0

    else:  # macro
        n_classes = len(labels)
        if n_classes > 0:
            precision = sum(m["precision"] for m in class_metrics.values()) / n_classes
            recall = sum(m["recall"] for m in class_metrics.values()) / n_classes
            f1 = sum(m["f1"] for m in class_metrics.values()) / n_classes
        else:
            precision = recall = f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_metrics": class_metrics,
    }


def compute_perplexity(
    model: Any,
    texts: List[str],
    tokenizer: Any = None,
    batch_size: int = 8,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Compute perplexity for a language model.

    Args:
        model: HuggingFace model
        texts: List of text samples
        tokenizer: Tokenizer (if None, will try to load from model)
        batch_size: Batch size for processing
        max_length: Maximum sequence length

    Returns:
        Dictionary with perplexity and loss
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot compute perplexity")
        return {"perplexity": float("inf"), "loss": float("inf")}

    if len(texts) == 0:
        return {"perplexity": float("inf"), "loss": float("inf")}

    try:
        from transformers import AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

        model.eval()
        device = next(model.parameters()).device

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encodings = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )

                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                # Get number of non-padded tokens
                n_tokens = attention_mask.sum().item()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        return {
            "perplexity": perplexity,
            "loss": avg_loss,
            "total_tokens": total_tokens,
        }

    except Exception as e:
        logger.error(f"Error computing perplexity: {e}")
        return {"perplexity": float("inf"), "loss": float("inf"), "error": str(e)}


def compute_exact_match(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute exact match score for QA tasks.

    Args:
        predictions: List of predicted answers
        references: List of reference answers
        normalize: Normalize text before comparison

    Returns:
        Dictionary with exact match score
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        return {"exact_match": 0.0, "matches": 0, "total": 0}

    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize:
            pred_norm = _normalize_answer(pred)
            ref_norm = _normalize_answer(ref)
        else:
            pred_norm = pred
            ref_norm = ref

        if pred_norm == ref_norm:
            matches += 1

    return {
        "exact_match": matches / len(predictions),
        "matches": matches,
        "total": len(predictions),
    }


def _normalize_answer(text: str) -> str:
    """Normalize answer for exact match comparison."""
    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    task_type: str = "generation",
) -> Dict[str, Any]:
    """
    Compute all relevant metrics for a given task type.

    Args:
        predictions: List of predicted outputs
        references: List of reference outputs
        task_type: Type of task ("generation", "classification", "qa")

    Returns:
        Dictionary with all computed metrics
    """
    results = {}

    if task_type == "generation":
        results["bleu"] = compute_bleu(predictions, references)
        results["rouge"] = compute_rouge(predictions, references)
        results["exact_match"] = compute_exact_match(predictions, references)

    elif task_type == "classification":
        results["accuracy"] = compute_accuracy(predictions, references)
        results["f1"] = compute_f1(predictions, references)

    elif task_type == "qa":
        results["exact_match"] = compute_exact_match(predictions, references)
        results["f1"] = compute_f1(
            [_tokenize(p) for p in predictions],
            [_tokenize(r) for r in references],
            average="micro",
        )

    else:
        # Default: compute all available metrics
        results["bleu"] = compute_bleu(predictions, references)
        results["rouge"] = compute_rouge(predictions, references)
        results["exact_match"] = compute_exact_match(predictions, references)

    return results
