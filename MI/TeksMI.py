import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# 1. Baca dataset
df = pd.read_csv('final.csv')

# 2. Identifikasi semua kolom skor (ground truth)
# Mengambil otomatis 10 kolom yang mengandung kata '(ground_truth)'
gt_cols = [col for col in df.columns if '(ground_truth)' in col]

# Tentukan kolom yang BUKAN fitur untuk di-drop dari X
non_features = ['graph', 'Question', 'Essay', 'image_number', 'Type'] + gt_cols

# 3. Siapkan fitur (X) - hanya ambil angka dan isi nilai kosong (NaN) dengan 0
X = df.drop(columns=non_features).select_dtypes(include=['number']).fillna(0)

# 4. Looping untuk menghitung MI pada setiap trait dan menyimpannya menjadi CSV
for gt in gt_cols:
    print(f"Sedang menghitung Mutual Information untuk: {gt}...")
    
    # Siapkan target (y)
    y = df[gt].fillna(df[gt].mean())
    
    # Hitung nilai MI
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Buat DataFrame hasil
    mi_df = pd.DataFrame({
        'Fitur': X.columns,
        'MI_Score': mi_scores
    })
    
    # Urutkan dari skor tertinggi ke terendah
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False).reset_index(drop=True)
    
    # Bersihkan nama trait untuk dijadikan nama file yang rapi
    trait_name = gt.replace('(ground_truth)', '').strip()
    filename = f"MI_Top_Features_{trait_name}.csv"
    
    # Simpan ke CSV
    mi_df.to_csv(filename, index=False)
    print(f"-> Selesai! File disimpan sebagai: {filename}\n")

print("Semua proses selesai! 10 File CSV berhasil dibuat.")