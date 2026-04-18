from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def validate_feature_dataset(
    input_rows: int,
    combined_dataset: pd.DataFrame,
    identity_columns: Iterable[str],
    target_columns: Iterable[str],
    expected_text_columns: Iterable[str],
    expected_image_columns: Iterable[str],
) -> Dict[str, object]:
    identity_columns = list(identity_columns)
    target_columns = list(target_columns)
    expected_text_columns = list(expected_text_columns)
    expected_image_columns = list(expected_image_columns)

    duplicate_columns = sorted(combined_dataset.columns[combined_dataset.columns.duplicated()].tolist())
    missing_targets = [col for col in target_columns if col not in combined_dataset.columns]
    missing_identity = [col for col in identity_columns if col not in combined_dataset.columns]

    txt_cols = [c for c in combined_dataset.columns if c.startswith("txt_")]
    img_cols = [c for c in combined_dataset.columns if c.startswith("img_")]
    feature_cols = txt_cols + img_cols

    non_numeric_feature_cols = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(combined_dataset[c])
    ]

    missing_expected_text = [col for col in expected_text_columns if col not in combined_dataset.columns]
    missing_expected_image = [col for col in expected_image_columns if col not in combined_dataset.columns]

    numeric_nan_ratio = (
        float(combined_dataset[feature_cols].isna().mean().mean()) if feature_cols else 0.0
    )

    report = {
        "input_rows": int(input_rows),
        "output_rows": int(len(combined_dataset)),
        "row_count_match": bool(len(combined_dataset) == input_rows),
        "duplicate_columns": duplicate_columns,
        "missing_target_columns": missing_targets,
        "missing_identity_columns": missing_identity,
        "missing_expected_text_columns": missing_expected_text,
        "missing_expected_image_columns": missing_expected_image,
        "text_feature_count": len(txt_cols),
        "image_feature_count": len(img_cols),
        "feature_count_total": len(feature_cols),
        "non_numeric_feature_columns": non_numeric_feature_cols,
        "feature_nan_ratio": numeric_nan_ratio,
        "expected_text_feature_count": len(expected_text_columns),
        "expected_image_feature_count": len(expected_image_columns),
        "is_valid_for_training": (
            len(duplicate_columns) == 0
            and len(missing_targets) == 0
            and len(missing_expected_text) == 0
            and len(missing_expected_image) == 0
            and len(non_numeric_feature_cols) == 0
            and len(combined_dataset) == input_rows
        ),
    }
    return report
