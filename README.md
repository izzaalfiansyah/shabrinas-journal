# Klasifikasi Kematangan Tapai — Perbandingan Model Machine Learning

Proyek ini membandingkan empat pendekatan machine learning untuk memprediksi status kematangan tapai berdasarkan data sensor suhu, kelembaban, dan kadar gas selama proses fermentasi.

---

## Dataset

**File:** `dataset/dataset_kematangan_tapai_v2.csv`

| Kolom | Deskripsi |
|-------|-----------|
| `percobaan` | Nomor percobaan (1–10) |
| `jam` | Waktu pengamatan (1–60 jam) |
| `suhu` | Suhu ruang fermentasi (°C) |
| `kelembaban` | Kelembaban relatif (%) |
| `kadar_gas` | Kadar gas fermentasi (ppm) |
| `status_kematangan` | Label: `belum matang`, `matang`, `terlalu matang` |

Total: **600 baris** data (10 percobaan × 60 jam).

---

## Struktur File

```
├── dataset/
│   └── dataset_kematangan_tapai_v2.csv
├── 1_linear_regression.ipynb   ← Logistic Regression
├── 2_svm.ipynb                 ← Support Vector Machine
├── 3_random_forest.ipynb       ← Random Forest
├── 4_rnn_lstm.ipynb            ← LSTM (Recurrent Neural Network)
└── README.md
```

---

## Penjelasan Masing-masing Model

### 1. Linear Regression → `1_linear_regression.ipynb`

**Library:** scikit-learn (`LogisticRegression`)

Model Logistic Regression adalah bentuk regresi linear yang diadaptasi untuk klasifikasi. Model ini belajar bobot (_coefficient_) untuk setiap fitur dan menghasilkan probabilitas kelas menggunakan fungsi softmax.

**Cara kerja:**
- Setiap baris data (satu observasi jam) diperlakukan sebagai sampel **independen**
- Model menemukan batas keputusan **linear** di ruang fitur
- Tidak ada memori antar waktu

**Kelebihan:**
- Sangat cepat dilatih
- Mudah diinterpretasi (koefisien per fitur)
- Baseline yang baik

**Keterbatasan:**
- Hanya mampu membuat batas keputusan linear
- Tidak mempertimbangkan urutan waktu fermentasi
- Cenderung underfitting pada data yang kompleks

---

### 2. Support Vector Machine → `2_svm.ipynb`

**Library:** scikit-learn (`SVC` dengan kernel RBF)

SVM bekerja dengan mencari **hyperplane optimal** yang memaksimalkan margin antar kelas. Dengan kernel RBF (Radial Basis Function), SVM dapat menangkap batas keputusan non-linear di ruang fitur berdimensi tinggi.

**Cara kerja:**
- Data ditransformasi ke ruang dimensi tinggi melalui fungsi kernel
- SVM mencari support vectors yang mendefinisikan batas kelas
- Setiap observasi jam tetap diperlakukan **independen**

**Kelebihan:**
- Efektif di ruang dimensi tinggi
- Robust terhadap outlier
- Mampu menangkap batas non-linear

**Keterbatasan:**
- Lambat pada dataset besar
- Tidak mempertimbangkan urutan temporal
- Membutuhkan tuning hyperparameter (C, gamma)

---

### 3. Random Forest → `3_random_forest.ipynb`

**Library:** scikit-learn (`RandomForestClassifier`)

Random Forest adalah metode ensemble yang membangun banyak decision tree secara paralel (dengan bootstrap sampling dan random feature selection), lalu menggabungkan prediksinya melalui voting mayoritas.

**Cara kerja:**
- Membangun 200 decision tree dari subset data dan fitur yang berbeda
- Tiap tree membuat prediksi, hasil akhir adalah voting mayoritas
- Memberikan skor **feature importance** yang berguna

**Kelebihan:**
- Akurasi tinggi tanpa perlu normalisasi data
- Robust terhadap overfitting
- Memberikan informasi feature importance
- Menangani hubungan non-linear dan interaksi fitur

**Keterbatasan:**
- Lebih lambat dibanding model linear
- Tidak mempertimbangkan urutan temporal
- Model sulit diinterpretasi secara keseluruhan

---

### 4. RNN/LSTM → `4_rnn_lstm.ipynb`

**Library:** TensorFlow/Keras (`LSTM`, `Bidirectional`)

LSTM (Long Short-Term Memory) adalah jenis Recurrent Neural Network yang dirancang khusus untuk data deret waktu. LSTM memiliki mekanisme **cell state** dan tiga **gates** (input, forget, output) yang memungkinkannya menyimpan dan "melupakan" informasi dari timestep sebelumnya secara selektif.

**Cara kerja:**
- Data dibentuk sebagai sequence: setiap sampel adalah jendela **10 jam berturut-turut**
- Format input: `[samples, timesteps=10, features=3]`
- Bidirectional LSTM memproses sequence dari dua arah
- Model belajar pola temporal kumulatif proses fermentasi

**Kelebihan:**
- Secara eksplisit memodelkan **dependensi temporal**
- Mampu menangkap pola jangka panjang (tren fermentasi)
- Paling sesuai secara ilmiah dengan proses fermentasi kumulatif
- Generalizes well ke percobaan baru

**Keterbatasan:**
- Membutuhkan lebih banyak data untuk training optimal
- Training lebih lambat dibanding model klasik
- Kurang interpretable (black box)

---

## Hasil Classification Report

### 1. Logistic Regression

```
Accuracy Score : 0.9031
F1 Score       : 0.9030

                precision    recall  f1-score   support

  belum matang       0.96      0.92      0.94       136
        matang       0.82      0.88      0.85       130
terlalu matang       0.93      0.90      0.92       147

      accuracy                           0.90       413
     macro avg       0.90      0.90      0.90       413
  weighted avg       0.91      0.90      0.90       413
```

### 2. Support Vector Machine (SVM)

```
Accuracy Score : 0.9128
F1 Score       : 0.9123

                precision    recall  f1-score   support

  belum matang       0.95      0.95      0.95       136
        matang       0.85      0.88      0.86       130
terlalu matang       0.94      0.90      0.92       147

      accuracy                           0.91       413
     macro avg       0.91      0.91      0.91       413
  weighted avg       0.91      0.91      0.91       413
```

### 3. Random Forest

```
Accuracy Score : 0.9370
F1 Score       : 0.9369

                precision    recall  f1-score   support

  belum matang       0.98      0.95      0.96       136
        matang       0.87      0.94      0.90       130
terlalu matang       0.96      0.93      0.94       147

      accuracy                           0.94       413
     macro avg       0.94      0.94      0.94       413
  weighted avg       0.94      0.94      0.94       413
```

### 4. RNN (LSTM)

```
Accuracy Score : 0.9646
F1 Score       : 0.9240

                precision    recall  f1-score   support

  belum matang       1.00      0.99      0.99       138
        matang       0.83      0.86      0.84        22
terlalu matang       0.92      0.95      0.94        38

      accuracy                           0.96       198
     macro avg       0.92      0.93      0.92       198
  weighted avg       0.97      0.96      0.97       198
```

---

## Perbandingan Model

| Aspek | Logistic Regression | SVM | Random Forest | LSTM (RNN) |
|-------|-------------------|-----|---------------|------------|
| **Jenis** | Linear | Kernel | Ensemble | Deep Learning |
| **Library** | scikit-learn | scikit-learn | scikit-learn | TensorFlow |
| **Accuracy** | 0.9031 | 0.9128 | 0.9370 | **0.9646** |
| **F1 Score (macro avg)** | 0.90 | 0.91 | 0.94 | **0.93** |
| **Memori temporal** | ❌ Tidak | ❌ Tidak | ❌ Tidak | ✅ Ya |
| **Dependensi antar jam** | ❌ Diabaikan | ❌ Diabaikan | ❌ Diabaikan | ✅ Dipelajari |
| **Non-linearity** | ❌ Tidak | ✅ Via kernel | ✅ Ya | ✅ Ya |
| **Kecepatan training** | ⚡ Sangat cepat | 🔸 Sedang | 🔸 Sedang | 🔴 Lambat |
| **Interpretabilitas** | ✅ Tinggi | 🔸 Sedang | ✅ Feature importance | ❌ Rendah |
| **Kebutuhan data** | Sedikit | Sedikit | Sedang | Banyak |
| **Cocok time series** | ❌ Tidak dirancang | ❌ Tidak dirancang | ❌ Tidak dirancang | ✅ Dirancang khusus |

---

## Mengapa RNN/LSTM Lebih Tepat untuk Dataset Ini?

> **Dataset kematangan tapai adalah data deret waktu murni.**

Proses fermentasi tapai adalah proses **kimia-biologis yang bersifat kumulatif dan berkelanjutan**:

1. **Kadar gas meningkat secara progresif** — produk fermentasi ragi yang terakumulasi dari jam ke jam
2. **Kelembaban meningkat bertahap** — hasil metabolisme mikroorganisme yang berlangsung terus-menerus  
3. **Status kematangan bersifat sekuensial** — tapai tidak bisa langsung "matang" tanpa melewati fase "belum matang"

Model klasik memperlakukan setiap observasi jam sebagai **sampel independen** — seolah jam ke-45 tidak ada hubungannya dengan jam ke-44. Ini bertentangan dengan realitas proses fermentasi.

**LSTM secara eksplisit memodelkan:**
- Tren kenaikan kadar gas dari waktu ke waktu
- Transisi bertahap antar fase kematangan
- Konteks kondisi beberapa jam sebelumnya untuk prediksi saat ini

Dengan demikian, meskipun model klasik mungkin mencapai akurasi yang kompetitif pada dataset ini (karena ada korelasi kuat antara fitur dengan label), **LSTM memberikan representasi yang lebih akurat secara ilmiah** dan lebih handal untuk generalisasi ke percobaan baru dengan kondisi berbeda.

---

## Cara Menjalankan

### Prasyarat

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### Menjalankan Notebook

```bash
jupyter notebook
```

Kemudian buka masing-masing notebook sesuai urutan:

1. `1_linear_regression.ipynb`
2. `2_svm.ipynb`
3. `3_random_forest.ipynb`
4. `4_rnn_lstm.ipynb`

> **Catatan:** Notebook 4 membutuhkan TensorFlow ≥ 2.10. Untuk training lebih cepat, gunakan GPU (opsional).

---

## Referensi

- Scikit-learn Documentation: https://scikit-learn.org/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
