import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_selection import mutual_info_regression
import warnings
import json
import sys
import os

warnings.filterwarnings('ignore')

# Opsional: Aktifkan GPU jika tersedia (Sesuai diskusi sebelumnya)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==========================================
# 0. FITUR AUTO-SAVE OUTPUT (LOGGING)
# ==========================================
class Logger(object):
    def __init__(self, filename="task1_training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

sys.stdout = Logger("task1_training_log.txt")

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("\n" + "="*60)
print("=== MEMULAI TASK: BAYESIAN TPE (BASED ON PAPER) + MI FILTER ===")
print("="*60)

# KUNCI: Pastikan Anda meload file data yang benar (final.csv)
# Jika dataset target terpisah, pastikan namanya sesuai.
X_raw = pd.read_csv('final.csv') 
df_targets = pd.read_csv('final.csv') 

trait_mapping = {
    'LA': 'lexical_accuracy(ground_truth)', 'LD': 'lexical_diversity(ground_truth)',
    'GA': 'grammatical_accuracy(ground_truth)', 'GD': 'grammatical_diversity(ground_truth)',
    'PA': 'punctuation_accuracy(ground_truth)', 'CH': 'coherence(ground_truth)',
    'OS': 'organizational_structure(ground_truth)', 'AC': 'argument_clarity(ground_truth)',
    'JP': 'justifying_persuasiveness(ground_truth)', 'EL': 'essay_length(ground_truth)'
}

print("Membersihkan data fitur...")
X_numeric = X_raw.select_dtypes(include=['number']).fillna(0)
# Membuang kolom identifier & target agar tidak bocor ke fitur latih
cols_to_drop = [col for col in X_numeric.columns if 'image_number' in col or '(ground_truth)' in col]
X_numeric = X_numeric.drop(columns=cols_to_drop, errors='ignore')

print(f"Total fitur awal yang siap dievaluasi: {X_numeric.shape[1]}\n")

# ==========================================
# 2. FUNGSI OBJEKTIF (Sesuai Paper & Anti-Overfitting)
# ==========================================
def objective(trial, X, y):
    # Parameter TPE yang SUDAH DISESUAIKAN (Anti-Overfitting & Based on Paper)
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "hist",       # Hist lebih cepat dan efisien
        "device": "cuda",            # Hapus atau ganti 'cpu' jika tidak pakai GPU
        "grow_policy": "lossguide",  # Wajib agar 'max_leaves' berfungsi
        "random_state": 42,
        "n_jobs": 1,
        
        # --- BAYESIAN TPE SEARCH SPACE ---
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 5),                # Dangkal (Anti-Overfitting)
        "max_leaves": trial.suggest_int("max_leaves", 7, 31),             # Selaras dengan max_depth
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),                  # Agresif memotong cabang tak berguna
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),          # Variasi sampel
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # Variasi fitur
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scores = []

    for fold, (train_index, valid_index) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # Latih menggunakan parameter dari Optuna
        model = xgb.XGBRegressor(**param, n_estimators=100)
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict(X_valid)
        
        # Evaluasi menggunakan QWK
        y_valid_int = np.round(y_valid * 2).astype(int)
        preds_int = np.clip(np.round(preds * 2), 0, 10).astype(int)
        
        qwk = cohen_kappa_score(y_valid_int, preds_int, weights='quadratic')
        qwk_scores.append(qwk)

    mean_qwk = np.mean(qwk_scores)
    # Print rapi untuk log
    print(f"  [Trial {trial.number}] depth:{param['max_depth']}, leaves:{param['max_leaves']}, lr:{param['learning_rate']:.3f} -> Mean QWK: {mean_qwk:.4f}")
    return mean_qwk

# ==========================================
# 3. PROSES TUNING OPTUNA
# ==========================================
best_params_per_trait = {}
N_TRIALS_TOTAL = 50  # 50 sudah cukup untuk ruang pencarian yang kecil ini
DB_NAME = "sqlite:///optuna_tpe_paper.db" 

print("Mulai Evaluasi per Trait...\n")

for trait_code, column_name in trait_mapping.items():
    if column_name not in df_targets.columns:
        print(f"  [SKIP] Kolom {column_name} tidak ditemukan.")
        continue

    print(f"======================================================")
    print(f" TRAIT: {trait_code} ({column_name})")
    print(f"======================================================")
    
    y_trait = df_targets[column_name].fillna(df_targets[column_name].mean())
    
    # -------------------------------------------------------------
    # MUTUAL INFORMATION FILTER
    # -------------------------------------------------------------
    print(f"  [>] Menjalankan seleksi fitur Mutual Information...")
    mi_scores = mutual_info_regression(X_numeric, y_trait, random_state=42)
    
    valid_features = X_numeric.columns[mi_scores > 0]
    X_filtered = X_numeric[valid_features]
    
    print(f"  [>] Fitur tersaring: Membuang {X_numeric.shape[1] - len(valid_features)} fitur tanpa informasi.")
    print(f"  [>] XGBoost akan dilatih dengan: {len(valid_features)} fitur relevan.\n")
    # -------------------------------------------------------------

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Gunakan TPESampler secara eksplisit
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=f"study_xgb_tpe_paper_{trait_code}", 
        storage=DB_NAME, 
        direction='maximize',
        sampler=sampler,
        load_if_exists=True 
    )
    
    trials_done = len(study.trials)
    if trials_done >= N_TRIALS_TOTAL:
        print(f"> Trait {trait_code} SUDAH SELESAI ({trials_done}/{N_TRIALS_TOTAL} trials). Skip...\n")
        best_params_per_trait[trait_code] = study.best_params
        continue
    
    trials_to_run = N_TRIALS_TOTAL - trials_done
    print(f"> Melanjutkan sisa {trials_to_run} trial dari total {N_TRIALS_TOTAL}...\n")
    
    study.optimize(lambda trial: objective(trial, X_filtered, y_trait), n_trials=trials_to_run)
    
    best_params_per_trait[trait_code] = study.best_params
    print(f"\n>>> HASIL AKHIR {trait_code} | Best QWK: {study.best_value:.4f}")
    print(f">>> Parameter Terbaik: {study.best_params}\n")

# ==========================================
# 4. SIMPAN HASIL HYPERPARAMETER
# ==========================================
with open('best_hyperparameters_tpe_paper.json', 'w') as f:
    json.dump(best_params_per_trait, f, indent=4)
print("\n[SUKSES] Semua trial selesai! File 'best_hyperparameters_tpe_paper.json' telah diperbarui!")