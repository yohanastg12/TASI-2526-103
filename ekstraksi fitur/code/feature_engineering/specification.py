from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FeatureSpec:
    feature_name: str
    modality: str
    source_columns: List[str]
    extractor_key: str
    output_columns: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    fallback_value: float = float("nan")
    paper_name: str = ""
    notes: str = ""


def _spec(
    feature_name: str,
    modality: str,
    source_columns: List[str],
    extractor_key: str,
    output_columns: List[str],
    params: Dict[str, Any] | None = None,
    fallback_value: float = float("nan"),
    paper_name: str = "",
    notes: str = "",
) -> FeatureSpec:
    return FeatureSpec(
        feature_name=feature_name,
        modality=modality,
        source_columns=source_columns,
        extractor_key=extractor_key,
        output_columns=output_columns,
        params=params or {},
        enabled=True,
        fallback_value=fallback_value,
        paper_name=paper_name,
        notes=notes,
    )


def get_default_feature_specs() -> List[FeatureSpec]:
    """Paper-based default spec: 12 text features + 3 image features."""
    text_specs = [
        _spec("topic_coherence", "text", ["Essay"], "topic_coherence", ["txt_topic_coherence"], paper_name="AWE 2025"),
        _spec("argument_density", "text", ["Essay"], "argument_density", ["txt_argument_density"], paper_name="AWE 2025"),
        _spec("sentence_complexity", "text", ["Essay"], "sentence_complexity", ["txt_sentence_complexity"], paper_name="AWE 2025"),
        _spec("dependency_depth", "text", ["Essay"], "dependency_depth", ["txt_dependency_depth"], paper_name="AWE 2025"),
        _spec("causal_connectives", "text", ["Essay"], "causal_connectives", ["txt_causal_connectives"], paper_name="AWE 2025"),
        _spec("contrast_markers", "text", ["Essay"], "contrast_markers", ["txt_contrast_markers"], paper_name="AWE 2025"),
        _spec("inference_indicators", "text", ["Essay"], "inference_indicators", ["txt_inference_indicators"], paper_name="AWE 2025"),
        _spec("paragraph_transitions", "text", ["Essay"], "paragraph_transitions", ["txt_paragraph_transitions"], paper_name="AWE 2025"),
        _spec("cohesion_chains", "text", ["Essay"], "cohesion_chains", ["txt_cohesion_chains"], paper_name="AWE 2025"),
        _spec("reference_resolution", "text", ["Essay"], "reference_resolution", ["txt_reference_resolution"], paper_name="AWE 2025"),
        _spec("vocabulary_diversity", "text", ["Essay"], "vocabulary_diversity", ["txt_vocabulary_diversity"], paper_name="AWE 2025"),
        _spec("academic_words", "text", ["Essay"], "academic_words", ["txt_academic_words"], paper_name="AWE 2025"),
    ]

    image_specs = [
        _spec(
            "box_detector_cascade_resnet50_fpn",
            "image",
            ["graph"],
            "box_detector_cascade_resnet50_fpn",
            ["img_box_detector_cascade_score"],
            paper_name="Towards an efficient framework for Data Extraction from Chart Images (2021)",
            notes="Proxy score inspired by Cascade R-CNN with ResNet-50 + FPN, RoIAlign 7x7, and NMS.",
        ),
        _spec(
            "point_detector_fcn_fusion",
            "image",
            ["graph"],
            "point_detector_fcn_fusion",
            ["img_point_detector_fcn_score"],
            paper_name="Towards an efficient framework for Data Extraction from Chart Images (2021)",
            notes="Proxy score inspired by FCN with multi-scale feature fusion and heatmap post-processing.",
        ),
        _spec(
            "legend_matching_embedding",
            "image",
            ["graph"],
            "legend_matching_embedding",
            ["img_legend_embedding_similarity"],
            paper_name="Towards an efficient framework for Data Extraction from Chart Images (2021)",
            notes="Similarity score from 128-D embedding style legend/element matching proxy.",
        ),
    ]

    return text_specs + image_specs


def split_specs_by_modality(specs: List[FeatureSpec]) -> Dict[str, List[FeatureSpec]]:
    grouped: Dict[str, List[FeatureSpec]] = {"text": [], "image": []}
    for spec in specs:
        if spec.modality in grouped and spec.enabled:
            grouped[spec.modality].append(spec)
    return grouped
