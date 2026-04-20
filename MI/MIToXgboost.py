import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("=== TAHAP FINAL: LOAD MI CSV -> XGBOOST FEATURE IMPORTANCE ===")
print("="*60)

# ==========================================
# 1. PERSIAPAN DATA UTAMA & PARAMETER
# ==========================================
# Load dataset utama (untuk mengambil nilai asli fitur dan skor esai)
df = pd.read_csv('final.csv')

# Load hyperparameter yang baru dituning
with open('best_hyperparameters_tpe_paper.json', 'r') as f:
    best_hp = json.load(f)

# Mapping trait ke nama file dan nama kolom target
trait_mapping = {
    'LA': 'lexical_accuracy', 'LD': 'lexical_diversity',
    'GA': 'grammatical_accuracy', 'GD': 'grammatical_diversity',
    'PA': 'punctuation_accuracy', 'CH': 'coherence',
    'OS': 'organizational_structure', 'AC': 'argument_clarity',
    'JP': 'justifying_persuasiveness', 'EL': 'essay_length'
}

# ==========================================
# 2. PROSES BACA CSV MI & TRAINING XGBOOST
# ==========================================
for trait_code, trait_name in trait_mapping.items():
    col_target = f"{trait_name}(ground_truth)"
    
    # Cek apakah kolom target ada di dataset
    if col_target not in df.columns:
        print(f"[SKIP] Kolom {col_target} tidak ada di dataset utama.")
        continue
        
    print(f"\nMemproses Trait: {trait_code} ({trait_name})")
    
    # --- TAHAP A: BACA FILE CSV MI ---
    # Nama file MI yang sudah di-generate sebelumnya
    mi_filename = f"MI_Top_Features_{trait_name}.csv"
    
    if not os.path.exists(mi_filename):
        print(f"  [ERROR] File {mi_filename} tidak ditemukan di folder! Melewati trait ini...")
        continue
        
    # Baca file CSV MI
    mi_df = pd.read_csv(mi_filename)
    
    # Ambil SEMUA fitur yang MI Score-nya > 0 (Sesuai instruksi Anda: tidak dibatasi)
    valid_mi_df = mi_df[mi_df['MI_Score'] > 0]
    valid_features = valid_mi_df['Fitur'].tolist()
    
    print(f"  -> Berhasil memuat {len(valid_features)} fitur (MI > 0) dari {mi_filename}.")
    
    # --- TAHAP B: SIAPKAN DATA UNTUK XGBOOST ---
    # Ambil label skor esai
    y = df[col_target].fillna(df[col_target].mean())
    
    # Ambil hanya kolom-kolom fitur yang lolos seleksi MI dari dataframe utama
    # (Pengecekan ekstra untuk memastikan nama kolom benar-benar ada di df)
    features_to_use = [f for f in valid_features if f in df.columns]
    X_filtered = df[features_to_use].fillna(0)
    
    # Ambil kembali skor MI dari valid_mi_df agar urutannya pas dengan features_to_use
    # untuk dicetak di hasil akhir
    mi_scores_to_save = valid_mi_df.set_index('Fitur').loc[features_to_use]['MI_Score'].values
    
    # --- TAHAP C: XGBOOST FEATURE IMPORTANCE ---
    params = best_hp.get(trait_code, {})
    if not params:
        print(f"  [WARNING] Parameter Optuna untuk {trait_code} kosong. Menggunakan parameter default XGBoost.")
        
    # Latih model XGBoost menggunakan parameter Optuna
    model = xgb.XGBRegressor(**params, n_estimators=100, random_state=42)
    model.fit(X_filtered, y)
    
    # Ekstrak nilai Importance (Gain) murni dari XGBoost
    xgb_importances = model.feature_importances_
    
    # --- TAHAP D: REKAPITULASI & SIMPAN ---
    # Gabungkan nama fitur, skor MI dari CSV, dan Importance dari XGBoost
    importance_df = pd.DataFrame({
        'Fitur': features_to_use,
        'MI_Score_Awal': mi_scores_to_save,
        'XGB_Importance_Gain': xgb_importances
    })
    
    # Urutkan berdasarkan fitur paling berpengaruh menurut XGBoost
    importance_df = importance_df.sort_values(by='XGB_Importance_Gain', ascending=False).reset_index(drop=True)
    
    # Simpan hasil akhir
    final_filename = f"Final_XGB_Importance_{trait_code}.csv"
    importance_df.to_csv(final_filename, index=False)
    
    # Cetak rangkuman
    top_1_fitur = importance_df.iloc[0]['Fitur']
    print(f"  -> File CSV tersimpan: {final_filename}")
    print(f"  -> ⭐ Fitur Paling Berpengaruh (Rank 1): {top_1_fitur}")

print("\n" + "="*60)
print("=== SELURUH PROSES SELESAI DENGAN SUKSES! ===")