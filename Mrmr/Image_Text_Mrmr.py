#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
TRAIT_MAPPING = {
    "LA": "lexical_accuracy",
    "LD": "lexical_diversity",
    "GA": "grammatical_accuracy",
    "GD": "grammatical_diversity",
    "PA": "punctuation_accuracy",
    "CH": "coherence",
    "OS": "organizational_structure",
    "AC": "argument_clarity",
    "JP": "justifying_persuasiveness",
    "EL": "essay_length",
}


def log_print(message: str, end: str = "\n") -> None:
    print(message, end=end)
    sys.stdout.flush()


def load_json(path: Path) -> Dict:
    if not path.exists():
        log_print(f"[WARNING] File JSON tidak ditemukan: {path}. Menggunakan parameter default.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset image-only tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset kosong: {path}")
    return df


def load_importance_features(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File importance tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if "Fitur" not in df.columns:
        raise ValueError(f"Kolom 'Fitur' tidak ditemukan di {path}")

    if "XGB_Importance_Gain" in df.columns:
        selected = df.loc[df["XGB_Importance_Gain"].fillna(0) > 0, "Fitur"].dropna().astype(str).tolist()
        if selected:
            return selected
    return df["Fitur"].dropna().astype(str).tolist()


def select_valid_numeric_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    valid = []
    for feat in features:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            valid.append(feat)
    return valid


def make_model_params(best_hp: Dict, trait_code: str) -> Dict:
    params = dict(best_hp.get(trait_code, {}))
    params.update({
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": -1,
    })
    return params


def run_training(
    image_csv: Path,
    hp_json: Path,
    importance_dir: Path,
    output_dir: Path,
) -> None:
    log_print("=" * 82)
    log_print("=== SCORING XGBOOST: SINGLE OUTPUT (IMAGE-ONLY | mRMR | CPU) ===")
    log_print("=" * 82)

    df = load_image_dataset(image_csv)
    log_print(f"[INFO] Berhasil meload image-only dataset: {image_csv}")

    best_hp = load_json(hp_json)
    if best_hp:
        log_print(f"[INFO] Berhasil meload hyperparameter: {hp_json}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "Trained_Single_Output_Models_mRMR_CPU_Gambar"
    model_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    final_results = []

    for trait_code, trait_name in TRAIT_MAPPING.items():
        col_target = f"{trait_name}(ground_truth)"
        if col_target not in df.columns:
            log_print(f"\n[TRAIT: {trait_code}] [SKIP] Kolom target {col_target} tidak ada di image.csv")
            continue

        importance_path = importance_dir / f"Final_XGB_Importance_mRMR_{trait_code}_gambar.csv"
        if not importance_path.exists():
            log_print(f"\n[TRAIT: {trait_code}] [SKIP] File importance tidak ditemukan: {importance_path}")
            continue

        raw_features = load_importance_features(importance_path)
        valid_features = select_valid_numeric_features(df, raw_features)

        if not valid_features:
            log_print(f"\n[TRAIT: {trait_code}] [SKIP] Tidak ada fitur numerik valid dari {importance_path.name}")
            continue

        log_print(f"\n[TRAIT: {trait_code}] Image-only mRMR untuk {trait_name}")
        log_print(f"    [INFO] Menggunakan {len(valid_features)} fitur gambar valid")

        X = df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = pd.to_numeric(df[col_target], errors="coerce").fillna(pd.to_numeric(df[col_target], errors="coerce").mean())

        params = make_model_params(best_hp, trait_code)
        fold_qwk = []
        fold_rmse = []

        for fold, (train_index, valid_index) in enumerate(kf.split(X), start=1):
            log_print(f"    [Fold {fold}/{N_SPLITS}] Training...", end=" ")
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            model = xgb.XGBRegressor(**params, n_estimators=100, random_state=RANDOM_STATE)
            model.fit(X_train, y_train)

            preds = model.predict(X_valid)
            rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
            fold_rmse.append(rmse)

            y_valid_int = np.round(y_valid * 2).astype(int)
            preds_int = np.clip(np.round(preds * 2), 0, 10).astype(int)
            qwk = float(cohen_kappa_score(y_valid_int, preds_int, weights="quadratic"))
            fold_qwk.append(qwk)
            log_print(f"OK (QWK: {qwk:.4f}, RMSE: {rmse:.4f})")

        mean_qwk = float(np.mean(fold_qwk))
        mean_rmse = float(np.mean(fold_rmse))
        final_results.append({
            "Trait": trait_code,
            "Trait_Name": trait_name,
            "Mean_QWK": mean_qwk,
            "Mean_RMSE": mean_rmse,
            "Num_Features": len(valid_features),
            "Importance_File": str(importance_path),
        })
        log_print(f"  ⭐ HASIL {trait_code} -> QWK: {mean_qwk:.4f} | RMSE: {mean_rmse:.4f}")

        final_model = xgb.XGBRegressor(**params, n_estimators=100, random_state=RANDOM_STATE)
        final_model.fit(X, y)
        model_path = model_dir / f"XGB_CPU_mRMR_{trait_code}_gambar.json"
        final_model.save_model(model_path)
        log_print(f"    [SAVE] Model final disimpan: {model_path}")

    if final_results:
        summary_path = output_dir / "Hasil_Scoring_mRMR_CPU_Gambar.csv"
        pd.DataFrame(final_results).to_csv(summary_path, index=False)
        log_print(f"\n[SYSTEM] Selesai. Rekap hasil disimpan di: {summary_path}")
    else:
        log_print("\n[SYSTEM] Gagal. Tidak ada model image-only mRMR yang berhasil dilatih.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training final XGBoost image-only mRMR untuk kebutuhan SHAP.")
    parser.add_argument(
        "--image-csv",
        type=Path,
        default=Path("/home/jovyan/work/TASI-103/YOTANG/FeatureImportance XGBoost/MI/image.csv"),
        help="Path ke image.csv",
    )
    parser.add_argument(
        "--hp-json",
        type=Path,
        default=Path("/home/jovyan/work/TASI-103/YOTANG/FeatureImportance XGBoost/MI/best_hyperparameters_tpe_paper.json"),
        help="Path ke best_hyperparameters_tpe_paper.json",
    )
    parser.add_argument(
        "--importance-dir",
        type=Path,
        default=Path("/home/jovyan/work/TASI-103/YOTANG/FeatureImportance XGBoost/MI"),
        help="Folder tempat Final_XGB_Importance_mRMR_*_gambar.csv berada",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jovyan/work/TASI-103/YOTANG/FeatureImportance XGBoost/MI/output_image"),
        help="Folder output model JSON dan rekap hasil",
    )
    args = parser.parse_args()
    run_training(
        image_csv=args.image_csv,
        hp_json=args.hp_json,
        importance_dir=args.importance_dir,
        output_dir=args.output_dir,
    )
