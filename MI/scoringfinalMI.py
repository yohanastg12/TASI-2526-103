import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

def log_print(message, end="\n"):
    print(message, end=end)
    sys.stdout.flush()

log_print("="*75)
log_print("=== SCORING XGBOOST: SINGLE OUTPUT (TEKS + GAMBAR | CPU MODE) ===")
log_print("="*75)

# ==========================================
# 1. PERSIAPAN DATA (MENGGABUNGKAN TEKS & GAMBAR)
# ==========================================
try:
    df_text = pd.read_csv('final.csv')
    log_print("[INFO] Berhasil meload final.csv (Fitur Teks)")
except FileNotFoundError:
    log_print("[ERROR] File final.csv tidak ditemukan di folder ini!")
    sys.exit()

try:
    # Menggunakan image.csv sesuai dengan penamaan pada script sebelumnya
    df_img = pd.read_csv('image.csv') 
    log_print("[INFO] Berhasil meload image.csv (Fitur Gambar)")
except FileNotFoundError:
    log_print("[ERROR] File image.csv tidak ditemukan di folder ini! (Coba cek apakah namanya gambar.csv)")
    sys.exit()

# Menggabungkan kedua dataset
common_cols = list(set(df_text.columns) & set(df_img.columns))
if 'image_number' in common_cols:
    df = pd.merge(df_text, df_img, on='image_number', suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.endswith('_drop')]
    log_print("[INFO] Dataset teks dan gambar berhasil digabungkan berdasarkan 'image_number'")
else:
    df = pd.concat([df_text, df_img], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    log_print("[INFO] Dataset teks dan gambar berhasil digabungkan secara berdampingan")

# Membaca parameter terbaik hasil Optuna versi Paper
try:
    with open('best_hyperparameters_tpe_paper.json', 'r') as f:
        best_hp = json.load(f)
except FileNotFoundError:
    log_print("[WARNING] File best_hyperparameters_tpe_paper.json tidak ditemukan. Menggunakan default.")
    best_hp = {}

trait_mapping = {
    'LA': 'lexical_accuracy', 'LD': 'lexical_diversity',
    'GA': 'grammatical_accuracy', 'GD': 'grammatical_diversity',
    'PA': 'punctuation_accuracy', 'CH': 'coherence',
    'OS': 'organizational_structure', 'AC': 'argument_clarity',
    'JP': 'justifying_persuasiveness', 'EL': 'essay_length'
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_results = []

# Buat folder khusus untuk model gabungan teks dan gambar
model_dir = "Trained_Single_Output_Models_CPU_Teks_Gambar"
os.makedirs(model_dir, exist_ok=True)

# ==========================================
# 2. PROSES TRAINING
# ==========================================
for trait_code, trait_name in trait_mapping.items():
    col_target = f"{trait_name}(ground_truth)"
    if col_target not in df.columns: 
        log_print(f"  [SKIP] Kolom target {col_target} tidak ada di dataset gabungan.")
        continue
        
    log_print(f"\n[TRAIT: {trait_code}] Menggunakan CPU untuk {trait_name} (Teks + Gambar)")
    
    # KUNCI PERBAIKAN: Load fitur dari Teks DAN Gambar
    fi_text_filename = f"Final_XGB_Importance_{trait_code}.csv"
    fi_img_filename = f"Final_XGB_Importance_{trait_code}_gambar.csv"
    
    selected_features = []
    
    # 2.1 Ambil fitur Teks
    if os.path.exists(fi_text_filename):
        df_fi_text = pd.read_csv(fi_text_filename)
        teks_features = df_fi_text[df_fi_text['XGB_Importance_Gain'] > 0]['Fitur'].tolist()
        selected_features.extend(teks_features)
    else:
        log_print(f"    [WARNING] File {fi_text_filename} tidak ditemukan.")
        
    # 2.2 Ambil fitur Gambar
    if os.path.exists(fi_img_filename):
        df_fi_img = pd.read_csv(fi_img_filename)
        img_features = df_fi_img[df_fi_img['XGB_Importance_Gain'] > 0]['Fitur'].tolist()
        selected_features.extend(img_features)
    else:
        log_print(f"    [WARNING] File {fi_img_filename} tidak ditemukan.")
        
    # 2.3 Filter dan Validasi Fitur
    valid_features = [f for f in selected_features if f in df.columns]
    
    if len(valid_features) == 0:
        log_print(f"    [SKIP] Tidak ada fitur valid (Gain > 0) untuk diproses pada trait ini.")
        continue
    elif len(valid_features) < len(selected_features):
        missing = len(selected_features) - len(valid_features)
        log_print(f"    [INFO] Menggunakan {len(valid_features)} fitur ({missing} fitur di-drop karena tidak ada di dataset).")
    else:
        log_print(f"    [INFO] Menggunakan total {len(valid_features)} fitur gabungan (Teks + Gambar).")
        
    # Siapkan data X dan y
    X = df[valid_features].fillna(0)
    y = df[col_target].fillna(df[col_target].mean())
    
    params = best_hp.get(trait_code, {}).copy()
    
    # --- KONFIGURASI CPU ---
    params.update({
        'tree_method': 'hist',
        'device': 'cpu',          
        'n_jobs': -1              
    })
    
    fold_qwk = []
    fold_rmse = []
    
    for fold, (train_index, valid_index) in enumerate(kf.split(X)):
        log_print(f"    [Fold {fold+1}/5] Training...", end=" ")
        
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # Training
        model = xgb.XGBRegressor(**params, n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        fold_rmse.append(rmse)
        
        y_valid_int = np.round(y_valid * 2).astype(int)
        preds_int = np.clip(np.round(preds * 2), 0, 10).astype(int)
        qwk = cohen_kappa_score(y_valid_int, preds_int, weights='quadratic')
        fold_qwk.append(qwk)
        
        log_print(f"OK (QWK: {qwk:.4f})")
        
    mean_qwk = np.mean(fold_qwk)
    mean_rmse = np.mean(fold_rmse)
    final_results.append({'Trait': trait_code, 'Mean_QWK': mean_qwk, 'Mean_RMSE': mean_rmse})
    
    log_print(f"  ⭐ HASIL {trait_code} -> QWK: {mean_qwk:.4f}")
    
    # Save model
    final_model = xgb.XGBRegressor(**params, n_estimators=100, random_state=42)
    final_model.fit(X, y)
    final_model.save_model(os.path.join(model_dir, f"XGB_CPU_{trait_code}_Teks_Gambar.json"))

# ==========================================
# 3. REKAP
# ==========================================
if final_results:
    rekap_df = pd.DataFrame(final_results)
    rekap_df.to_csv('Hasil_Scoring_CPU_Teks_Gambar.csv', index=False)
    log_print("\n[SYSTEM] Selesai! Hasil direkap di 'Hasil_Scoring_CPU_Teks_Gambar.csv'")
else:
    log_print("\n[SYSTEM] Gagal. Tidak ada model yang dilatih.")