from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .specification import FeatureSpec, get_default_feature_specs

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?;。\n]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "they", "their", "them",
    "he", "his", "she", "her", "we", "our", "you", "your", "i", "my",
}

_ACADEMIC_LEXICON = {
    "analysis", "argument", "evidence", "reasoning", "conclusion", "concept",
    "principle", "hypothesis", "theory", "significant", "relevant", "logical",
    "coherence", "inference", "assumption", "evaluate", "critical", "perspective",
}

_CAUSAL_MARKERS = [
    "because", "since", "therefore", "thus", "hence", "as a result",
    "reduces", "impairs", "can improve", "leads to", "causes",
]

_CONTRAST_MARKERS = ["however", "although", "on the contrary", "while", "but", "yet", "nevertheless"]

_INFERENCE_MARKERS = [
    "therefore", "thus", "in conclusion", "this means", "which implies", "suggesting that",
]

_TRANSITION_MARKERS = [
    "first", "firstly", "second", "secondly", "finally",
    "however", "therefore", "in summary", "moreover", "furthermore",
]

_COMPLEXITY_MARKERS = [
    "because", "since", "although", "however", "therefore", "which",
    "that", "while", "if", "when", "whereas", "thus",
]

_ARGUMENT_MARKERS = [
    "because", "therefore", "thus", "hence", "for example", "for instance",
    "however", "although", "in conclusion", "this means", "which implies", "suggesting that",
]


ExtractorFunction = Callable[[Mapping[str, Any], Dict[str, Any]], float]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _to_tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(_to_text(text))]


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_PATTERN.split(_to_text(text))
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    raw = _to_text(text)
    parts = re.split(r"\n\s*\n", raw)
    return [p.strip() for p in parts if p.strip()]


def _count_phrase_occurrences(text: str, phrases: Iterable[str]) -> int:
    text_low = _to_text(text).lower()
    return sum(text_low.count(phrase.lower()) for phrase in phrases)


def _content_tokens(text: str) -> set[str]:
    return {token for token in _to_tokens(text) if token not in _STOPWORDS and len(token) > 2}


def _frequency_per_100_tokens(text: str, markers: Iterable[str]) -> float:
    tokens = _to_tokens(text)
    return 100.0 * _safe_div(_count_phrase_occurrences(text, markers), len(tokens))


class TextFeatureExtractor:
    """Extract exactly 12 text features from paper.txt/def.txt."""

    def __init__(self, specs: Sequence[FeatureSpec] | None = None):
        all_specs = list(specs) if specs is not None else get_default_feature_specs()
        self.specs = [spec for spec in all_specs if spec.modality == "text" and spec.enabled]
        self.registry: Dict[str, ExtractorFunction] = {
            "topic_coherence": self._topic_coherence,
            "argument_density": self._argument_density,
            "sentence_complexity": self._sentence_complexity,
            "dependency_depth": self._dependency_depth,
            "causal_connectives": self._causal_connectives,
            "contrast_markers": self._contrast_markers,
            "inference_indicators": self._inference_indicators,
            "paragraph_transitions": self._paragraph_transitions,
            "cohesion_chains": self._cohesion_chains,
            "reference_resolution": self._reference_resolution,
            "vocabulary_diversity": self._vocabulary_diversity,
            "academic_words": self._academic_words,
        }

    def expected_feature_columns(self) -> List[str]:
        cols: List[str] = []
        for spec in self.specs:
            cols.extend(spec.output_columns)
        return cols

    def extract(self, row: Mapping[str, Any]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for spec in self.specs:
            fallback_value = float(spec.fallback_value)
            col = spec.output_columns[0]
            func = self.registry.get(spec.extractor_key)

            if func is None:
                features[col] = fallback_value
                continue

            try:
                features[col] = float(func(row, spec.params))
            except Exception:
                features[col] = fallback_value

        return features

    @staticmethod
    def _essay(row: Mapping[str, Any]) -> str:
        return _to_text(row.get("Essay", ""))

    def _topic_coherence(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        """Proxy for embedding coherence using adjacent sentence lexical overlap."""
        sentences = _split_sentences(self._essay(row))
        if len(sentences) < 2:
            return 0.0

        overlaps: List[float] = []
        for s1, s2 in zip(sentences[:-1], sentences[1:]):
            t1 = _content_tokens(s1)
            t2 = _content_tokens(s2)
            overlaps.append(_safe_div(len(t1 & t2), len(t1 | t2)))
        return float(np.mean(overlaps)) if overlaps else 0.0

    def _argument_density(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        """Argument marker occurrences per 100 characters."""
        essay = self._essay(row)
        marker_hits = _count_phrase_occurrences(essay, _ARGUMENT_MARKERS)
        return 100.0 * _safe_div(marker_hits, len(essay))

    def _sentence_complexity(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        essay = self._essay(row)
        sentences = _split_sentences(essay)
        if not sentences:
            return 0.0

        tokens_per_sentence = [len(_to_tokens(sentence)) for sentence in sentences]
        avg_len = float(np.mean(tokens_per_sentence))
        marker_density = _safe_div(_count_phrase_occurrences(essay, _COMPLEXITY_MARKERS), len(sentences))
        separator_density = _safe_div(len(re.findall(r"[,;:]", essay)), len(sentences))
        return float(avg_len + 1.5 * marker_density + 0.5 * separator_density)

    def _dependency_depth(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        """Dependency depth proxy without external parser."""
        essay = self._essay(row)
        sentences = _split_sentences(essay)
        if not sentences:
            return 0.0

        subordinator_hits = _count_phrase_occurrences(essay, _COMPLEXITY_MARKERS)
        separator_hits = len(re.findall(r"[,;:()]", essay))
        return _safe_div(subordinator_hits + separator_hits, len(sentences))

    def _causal_connectives(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        return _frequency_per_100_tokens(self._essay(row), _CAUSAL_MARKERS)

    def _contrast_markers(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        return _frequency_per_100_tokens(self._essay(row), _CONTRAST_MARKERS)

    def _inference_indicators(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        return _frequency_per_100_tokens(self._essay(row), _INFERENCE_MARKERS)

    def _paragraph_transitions(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        paragraphs = _split_paragraphs(self._essay(row))
        if len(paragraphs) <= 1:
            return 0.0

        transition_hits = 0
        for paragraph in paragraphs[1:]:
            first_tokens = " ".join(_to_tokens(paragraph)[:5])
            if any(marker in first_tokens for marker in _TRANSITION_MARKERS):
                transition_hits += 1
        return _safe_div(transition_hits, len(paragraphs) - 1)

    def _cohesion_chains(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        essay = self._essay(row)
        sentences = _split_sentences(essay)
        if len(sentences) < 2:
            return 0.0

        overlaps: List[float] = []
        for s1, s2 in zip(sentences[:-1], sentences[1:]):
            t1 = _content_tokens(s1)
            t2 = _content_tokens(s2)
            overlaps.append(_safe_div(len(t1 & t2), len(t1 | t2)))

        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        connective_density = _frequency_per_100_tokens(essay, _CAUSAL_MARKERS + _CONTRAST_MARKERS + _INFERENCE_MARKERS)
        connective_score = min(connective_density / 10.0, 1.0)
        return float(0.7 * mean_overlap + 0.3 * connective_score)

    def _reference_resolution(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        sentences = _split_sentences(self._essay(row))
        if not sentences:
            return 0.0

        pronouns = {"this", "it", "its", "they", "their", "them", "these", "those", "he", "she", "his", "her"}
        total_refs = 0
        resolved_refs = 0
        context_window: deque[set[str]] = deque(maxlen=2)

        for sentence in sentences:
            tokens = _to_tokens(sentence)
            current_content = _content_tokens(sentence)
            context_content = set().union(*context_window) if context_window else set()

            for token in tokens:
                if token in pronouns:
                    total_refs += 1
                    if context_content:
                        resolved_refs += 1

            context_window.append(current_content)

        return _safe_div(resolved_refs, total_refs)

    def _vocabulary_diversity(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        tokens = _to_tokens(self._essay(row))
        return _safe_div(len(set(tokens)), len(tokens))

    def _academic_words(self, row: Mapping[str, Any], _: Dict[str, Any]) -> float:
        tokens = _to_tokens(self._essay(row))
        if not tokens:
            return 0.0
        academic_hits = sum(1 for token in tokens if token in _ACADEMIC_LEXICON)
        return _safe_div(academic_hits, len(tokens))
