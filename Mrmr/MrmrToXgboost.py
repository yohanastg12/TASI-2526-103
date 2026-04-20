import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("=== TAHAP FINAL: LOAD MRMR CSV -> XGBOOST FEATURE IMPORTANCE ===")
print("="*60)

# ==========================================
# 1. PERSIAPAN DATA UTAMA & PARAMETER
# ==========================================
# Load dataset utama (final.csv)
df = pd.read_csv('final.csv')

# Load hyperparameter hasil Optuna
with open('best_hyperparameters_tpe_paper.json', 'r') as f:
    best_hp = json.load(f)

# Mapping trait ke nama kolom target
trait_mapping = {
    'LA': 'lexical_accuracy', 'LD': 'lexical_diversity',
    'GA': 'grammatical_accuracy', 'GD': 'grammatical_diversity',
    'PA': 'punctuation_accuracy', 'CH': 'coherence',
    'OS': 'organizational_structure', 'AC': 'argument_clarity',
    'JP': 'justifying_persuasiveness', 'EL': 'essay_length'
}

# ==========================================
# 2. PROSES BACA CSV MRMR & TRAINING XGBOOST
# ==========================================
for trait_code, trait_name in trait_mapping.items():
    col_target = f"{trait_name}(ground_truth)"
    
    if col_target not in df.columns:
        print(f"[SKIP] Kolom {col_target} tidak ada di dataset utama.")
        continue
        
    print(f"\nMemproses Trait: {trait_code} ({trait_name})")
    
    # --- TAHAP A: BACA FILE CSV MRMR ---
    # Berdasarkan file yang Anda unggah: mrmr_all_argument_clarity(ground_truth).csv
    mrmr_filename = f"mrmr_all_{col_target}.csv"
    
    # Coba cari di folder saat ini, atau di sub-folder tempat script mRMR menyimpannya
    if not os.path.exists(mrmr_filename):
        fallback_path = os.path.join("output_mrmr", "mrmr", "rankings", mrmr_filename)
        if os.path.exists(fallback_path):
            mrmr_filename = fallback_path
        else:
            print(f"  [ERROR] File {mrmr_filename} tidak ditemukan! Melewati trait ini...")
            continue
            
    mrmr_df = pd.read_csv(mrmr_filename)
    
    # KUNCI PERUBAHAN: Jangan filter `> 0` karena mRMR score bisa minus! 
    # Ambil SEMUA fitur yang sudah diranking
    valid_features = mrmr_df['Fitur'].tolist()
    
    print(f"  -> Berhasil memuat {len(valid_features)} fitur dari {mrmr_filename}.")
    
    # --- TAHAP B: SIAPKAN DATA UNTUK XGBOOST ---
    y = df[col_target].fillna(df[col_target].mean())
    
    features_to_use = [f for f in valid_features if f in df.columns]
    X_filtered = df[features_to_use].fillna(0)
    
    # Ambil skor mRMR dari CSV agar urutannya pas dengan fitur untuk hasil akhir
    mrmr_scores_to_save = mrmr_df.set_index('Fitur').loc[features_to_use]['mRMR_Score'].values
    
    # --- TAHAP C: XGBOOST FEATURE IMPORTANCE ---
    params = best_hp.get(trait_code, {})
    if not params:
        print(f"  [WARNING] Parameter Optuna untuk {trait_code} kosong. Menggunakan parameter default XGBoost.")
        
    # Tambahkan parameter teknis (menggunakan CPU agar aman dari error device)
    params.update({'n_estimators': 100, 'tree_method': 'hist', 'device': 'cpu', 'random_state': 42})
        
    model = xgb.XGBRegressor(**params)
    model.fit(X_filtered, y)
    
    # Ekstrak nilai Importance (Gain) murni dari XGBoost
    xgb_importances = model.feature_importances_
    
    # --- TAHAP D: REKAPITULASI & SIMPAN ---
    importance_df = pd.DataFrame({
        'Fitur': features_to_use,
        'mRMR_Score_Awal': mrmr_scores_to_save,
        'XGB_Importance_Gain': xgb_importances
    })
    
    # Urutkan berdasarkan fitur paling berpengaruh menurut XGBoost (Gain tertinggi)
    importance_df = importance_df.sort_values(by='XGB_Importance_Gain', ascending=False).reset_index(drop=True)
    
    # Simpan dengan nama spesifik untuk mRMR agar tidak menimpa hasil MI sebelumnya
    final_filename = f"Final_XGB_Importance_mRMR_{trait_code}.csv"
    importance_df.to_csv(final_filename, index=False)
    
    top_1_fitur = importance_df.iloc[0]['Fitur']
    print(f"  -> File CSV tersimpan: {final_filename}")
    print(f"  -> ⭐ Fitur Paling Berpengaruh (Rank 1): {top_1_fitur}")

print("\n" + "="*60)
print("=== SELURUH PROSES SELESAI DENGAN SUKSES! ===")