from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.constants import IDENTITY_COLUMNS, RAW_REQUIRED_COLUMNS, TARGET_COLUMNS
from feature_engineering.image_features import ImageFeatureExtractor
from feature_engineering.specification import get_default_feature_specs, split_specs_by_modality
from feature_engineering.text_features import TextFeatureExtractor
from feature_engineering.validators import validate_feature_dataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> Dict[str, Path]:
    root = _project_root()
    dataset_dir = root / "dataset"
    output_dir = root / "output"
    return {
        "input_csv": dataset_dir / "data.csv",
        "output_dataset_csv": output_dir / "dataset.csv",
        "output_text_csv": output_dir / "text.csv",
        "output_image_csv": output_dir / "image.csv",
        "error_log_csv": output_dir / "feature_extraction_errors.csv",
        "summary_json": output_dir / "feature_extraction_summary.json",
    }


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()

    parser = argparse.ArgumentParser(description="Build multimodal feature dataset for AES research")
    parser.add_argument("--input-csv", type=str, default=str(defaults["input_csv"]))
    parser.add_argument("--output-dataset-csv", type=str, default=str(defaults["output_dataset_csv"]))
    parser.add_argument("--output-text-csv", type=str, default=str(defaults["output_text_csv"]))
    parser.add_argument("--output-image-csv", type=str, default=str(defaults["output_image_csv"]))
    parser.add_argument("--error-log-csv", type=str, default=str(defaults["error_log_csv"]))
    parser.add_argument("--summary-json", type=str, default=str(defaults["summary_json"]))
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--strict", action="store_true", help="Stop on first row error")
    return parser.parse_args()


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")


def _read_input_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Failed to read CSV with tried encodings {encodings}: {last_error}")


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ordered_columns(df_out: pd.DataFrame) -> List[str]:
    txt_cols = sorted([c for c in df_out.columns if c.startswith("txt_")])
    img_cols = sorted([c for c in df_out.columns if c.startswith("img_")])

    ordered = []
    ordered.extend([c for c in IDENTITY_COLUMNS if c in df_out.columns])
    ordered.extend(txt_cols)
    ordered.extend(img_cols)
    ordered.extend([c for c in TARGET_COLUMNS if c in df_out.columns])

    for col in df_out.columns:
        if col not in ordered:
            ordered.append(col)

    return ordered


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dataset_csv = Path(args.output_dataset_csv)
    output_text_csv = Path(args.output_text_csv)
    output_image_csv = Path(args.output_image_csv)
    error_log_csv = Path(args.error_log_csv)
    summary_json = Path(args.summary_json)

    output_dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    output_text_csv.parent.mkdir(parents=True, exist_ok=True)
    output_image_csv.parent.mkdir(parents=True, exist_ok=True)
    error_log_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading input dataset: {input_csv}")
    df = _read_input_csv(input_csv)
    _check_required_columns(df)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
        print(f"[INFO] Running on subset rows: {len(df)}")

    specs = get_default_feature_specs()
    grouped_specs = split_specs_by_modality(specs)

    text_extractor = TextFeatureExtractor(specs=grouped_specs.get("text", []))
    image_extractor = ImageFeatureExtractor(
        specs=grouped_specs.get("image", []),
        image_root=args.image_root or None,
    )

    expected_text_cols = text_extractor.expected_feature_columns()
    expected_image_cols = image_extractor.expected_feature_columns()

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"[INFO] Start feature extraction for {len(df)} rows")

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        base_record: Dict[str, Any] = {
            **{col: row_dict.get(col, np.nan) for col in IDENTITY_COLUMNS},
            **{col: row_dict.get(col, np.nan) for col in TARGET_COLUMNS},
        }

        text_feats: Dict[str, Any]
        image_feats: Dict[str, Any]

        try:
            text_feats = text_extractor.extract(row_dict)
            if not str(row_dict.get("Essay", "") or "").strip():
                errors.append(
                    {
                        "row_index": int(idx),
                        "stage": "text",
                        "error_type": "EMPTY_ESSAY",
                        "message": "Essay is empty. Text features use fallback values.",
                    }
                )
        except Exception as exc:
            text_feats = {col: float("nan") for col in expected_text_cols}
            errors.append(
                {
                    "row_index": int(idx),
                    "stage": "text",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=1),
                }
            )
            if args.strict:
                raise

        try:
            image_feats = image_extractor.extract(row_dict)
            image_all_nan = all(pd.isna(image_feats.get(col, np.nan)) for col in expected_image_cols)
            if image_all_nan:
                errors.append(
                    {
                        "row_index": int(idx),
                        "stage": "image",
                        "error_type": "IMAGE_LOAD_FAILED",
                        "message": f"Image not readable from graph={row_dict.get('graph', '')}",
                    }
                )
        except Exception as exc:
            image_feats = {col: float("nan") for col in expected_image_cols}
            errors.append(
                {
                    "row_index": int(idx),
                    "stage": "image",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=1),
                }
            )
            if args.strict:
                raise

        record = {**base_record, **text_feats, **image_feats}
        records.append(record)

        if (idx + 1) % 100 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(df)} rows")

    df_out = pd.DataFrame(records)

    for col in expected_text_cols + expected_image_cols:
        if col not in df_out.columns:
            df_out[col] = float("nan")

    ordered_cols = _ordered_columns(df_out)
    df_out = df_out[ordered_cols]

    validation_report = validate_feature_dataset(
        input_rows=len(df),
        combined_dataset=df_out,
        identity_columns=IDENTITY_COLUMNS,
        target_columns=TARGET_COLUMNS,
        expected_text_columns=expected_text_cols,
        expected_image_columns=expected_image_cols,
    )

    df_out.to_csv(output_dataset_csv, index=False)

    text_cols = [c for c in df_out.columns if c.startswith("txt_")]
    image_cols = [c for c in df_out.columns if c.startswith("img_")]

    text_df = df_out[[*IDENTITY_COLUMNS, *text_cols, *[c for c in TARGET_COLUMNS if c in df_out.columns]]]
    image_df = df_out[[*IDENTITY_COLUMNS, *image_cols, *[c for c in TARGET_COLUMNS if c in df_out.columns]]]

    text_df.to_csv(output_text_csv, index=False)
    image_df.to_csv(output_image_csv, index=False)

    error_df = pd.DataFrame(errors)
    if not error_df.empty:
        error_df.to_csv(error_log_csv, index=False)
    else:
        pd.DataFrame(columns=["row_index", "stage", "error_type", "message", "traceback"]).to_csv(
            error_log_csv, index=False
        )

    summary_payload = {
        "input_csv": str(input_csv),
        "output_dataset_csv": str(output_dataset_csv),
        "output_text_csv": str(output_text_csv),
        "output_image_csv": str(output_image_csv),
        "error_log_csv": str(error_log_csv),
        "rows_input": int(len(df)),
        "rows_output": int(len(df_out)),
        "error_count": int(len(errors)),
        "text_feature_count": int(len([c for c in df_out.columns if c.startswith("txt_")])),
        "image_feature_count": int(len([c for c in df_out.columns if c.startswith("img_")])),
        "validation": {k: _safe_serialize(v) for k, v in validation_report.items()},
    }

    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print(f"[INFO] Combined dataset saved: {output_dataset_csv}")
    print(f"[INFO] Text-only dataset saved: {output_text_csv}")
    print(f"[INFO] Image-only dataset saved: {output_image_csv}")
    print(f"[INFO] Error log saved: {error_log_csv}")
    print(f"[INFO] Summary saved: {summary_json}")
    print(f"[INFO] Validation status is_valid_for_training={validation_report['is_valid_for_training']}")


if __name__ == "__main__":
    main()
