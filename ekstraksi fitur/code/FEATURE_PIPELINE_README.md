Feature Engineering Pipeline (Multimodal AES)

1) Tujuan
- Mengubah dataset mentah pasangan essay-gambar menjadi dataset fitur numerik siap training.
- Menjaga kolom target ground truth tetap utuh.
- Menggunakan tepat 15 fitur berbasis paper/info:
  - 12 fitur teks (paper AWE 2025)
  - 3 fitur gambar (Box detector, Point detector, Legend matching)

2) Struktur
- build_feature_dataset.py: runner utama.
- feature_engineering/text_features.py: extractor fitur teks.
- feature_engineering/image_features.py: extractor fitur gambar.
- feature_engineering/specification.py: spesifikasi default 15 fitur (tanpa Excel).
- feature_engineering/validators.py: validasi output.

3) Cara jalan
- Jalankan dari folder code:
  python build_feature_dataset.py
- Backend image extractor default: `deep` (torch/torchvision). Jika ingin pakai fallback proxy, set env `AES_IMAGE_BACKEND=proxy`.

4) Output
- output/dataset.csv (gabungan: identity + text + image + target)
- output/text.csv (identity + text + target)
- output/image.csv (identity + image + target)
- output/feature_extraction_errors.csv
- output/feature_extraction_summary.json

5) Kompatibilitas training script
- Kolom graph, Question, Essay, image_number, Type tetap dipertahankan.
- Semua fitur hasil ekstraksi diberi prefix txt_ atau img_.
- Target columns argument_clarity(ground_truth) dst tetap dipertahankan.
- Format ini kompatibel dengan pola:
  X = df.drop(columns=DROP_COLUMNS + targets).select_dtypes(include=[np.number])
  Y = df[present_targets]
