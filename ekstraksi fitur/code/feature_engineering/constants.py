from __future__ import annotations

IDENTITY_COLUMNS = [
    "graph",
    "Question",
    "Essay",
    "image_number",
    "Type",
]

TARGET_COLUMNS = [
    "argument_clarity(ground_truth)",
    "justifying_persuasiveness(ground_truth)",
    "organizational_structure(ground_truth)",
    "coherence(ground_truth)",
    "essay_length(ground_truth)",
    "grammatical_accuracy(ground_truth)",
    "grammatical_diversity(ground_truth)",
    "lexical_accuracy(ground_truth)",
    "lexical_diversity(ground_truth)",
    "punctuation_accuracy(ground_truth)",
]

RAW_REQUIRED_COLUMNS = IDENTITY_COLUMNS + TARGET_COLUMNS

FEATURE_SPEC_COLUMNS = [
    "feature_name",
    "modality",
    "source_columns",
    "extractor_key",
    "output_columns",
    "params_json",
    "enabled",
    "fallback_value",
    "paper_name",
    "code_snippet",
    "notes",
]
