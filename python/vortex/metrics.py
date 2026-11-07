"""Explicit ranking metrics and limited lexical answer diagnostics."""
import math
import re
from collections import Counter


def retrieval_metrics(retrieved, relevant, k, grades=None):
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    ranked = list(retrieved)
    if len(set(ranked)) != len(ranked):
        raise ValueError("retrieved IDs must be unique")
    labels = set(relevant)
    if not labels:
        raise ValueError("retrieval evaluation requires at least one relevant ID")
    if grades is None:
        gains = {identifier: 1.0 for identifier in labels}
    else:
        gains = {int(identifier): float(value) for identifier, value in grades.items()}
        if any(not math.isfinite(value) or value < 0 or value > 30 for value in gains.values()):
            raise ValueError("relevance grades must be finite and between 0 and 30")
        if {identifier for identifier, value in gains.items() if value > 0} != labels:
            raise ValueError("positive graded IDs must match relevant IDs")
    top = ranked[:k]
    found = [rank for rank, identifier in enumerate(top, 1) if identifier in labels]
    dcg = sum((2 ** gains.get(identifier, 0) - 1) / math.log2(rank + 1)
              for rank, identifier in enumerate(top, 1))
    ideal = sum((2 ** value - 1) / math.log2(rank + 1)
                for rank, value in enumerate(sorted(gains.values(), reverse=True)[:k], 1))
    return {"precision": len(found) / k, "recall": len(found) / len(labels),
            "hit": float(bool(found)), "mrr": 1 / found[0] if found else 0.0,
            "ndcg": dcg / ideal if ideal else 0.0}


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def answer_metrics(answer, reference, context):
    """Conservative quotation checks and lexical overlap; no semantic entailment claim."""
    actual = tokens(answer)
    expected = tokens(reference) if reference is not None else None
    overlap = sum((Counter(actual) & Counter(expected or [])).values())
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    context_tokens = set(tokens(context))
    split = r"(?<=[.!?])\s+|\n+"
    claims = [tokens(sentence) for sentence in re.split(split,answer) if tokens(sentence)]
    evidence = [tokens(sentence) for sentence in re.split(split,context) if tokens(sentence)]
    supported = [any(claim == sentence[start:start+len(claim)]
                     for sentence in evidence for start in range(len(sentence)-len(claim)+1))
                 for claim in claims]
    return {
        "exact_match": float(actual == expected) if expected is not None else None,
        "token_f1": (2 * precision * recall / (precision + recall)
                     if precision + recall else float(actual == expected)) if expected is not None else None,
        "lexical_support": sum(token in context_tokens for token in actual) / len(actual) if actual else None,
        "verbatim_sentence_support": sum(supported)/len(supported) if supported else None,
    }
