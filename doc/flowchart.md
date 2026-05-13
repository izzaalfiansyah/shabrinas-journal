# Flowchart Implementasi Training — Prediksi Kematangan Tapai Singkong

Dokumen ini menggambarkan alur implementasi training untuk keempat model machine learning yang digunakan dalam klasifikasi kematangan tapai singkong berdasarkan data sensor suhu, kelembaban, dan kadar gas fermentasi.

---

## 1. Alur Umum (Overview)

```mermaid
flowchart TD
    A([🟢 Mulai]) --> B[Load Dataset\ndataset_kematangan_tapai.csv]
    B --> C[Eksplorasi Data\nEDA & Statistik Deskriptif]
    C --> D{Data\nBersih?}
    D -- Tidak --> E[Preprocessing\nHandling Missing Values\nRemove Duplicates]
    E --> F[Feature Engineering\nsuhu · kelembaban · kadar_gas]
    D -- Ya --> F
    F --> G[Encode Label\nbelum matang · matang · terlalu matang]
    G --> H[Split Data\nTrain 69% · Test 31%]
    H --> I{Pilih\nModel}
    I --> J[Logistic\nRegression]
    I --> K[Support Vector\nMachine]
    I --> L[Random\nForest]
    I --> M[RNN / LSTM]
    J --> N[Evaluasi Model]
    K --> N
    L --> N
    M --> N
    N --> O[Bandingkan\nAccuracy & F1 Score]
    O --> P([🔴 Selesai])
```

---

## 2. Preprocessing & Persiapan Data

```mermaid
flowchart TD
    A([Mulai Preprocessing]) --> B[Baca CSV\npd.read_csv]
    B --> C[Cek Info Dataset\ndf.info · df.describe]
    C --> D[Cek Missing Values\ndf.isnull().sum]
    D --> E{Ada\nMissing\nValues?}
    E -- Ya --> F[Isi atau Drop\nMissing Values]
    E -- Tidak --> G[Pilih Fitur\nsuhu · kelembaban · kadar_gas]
    F --> G
    G --> H[Encode Label Target\nLabelEncoder\nbelum matang=0 · matang=1 · terlalu matang=2]
    H --> I[Normalisasi Fitur\nStandardScaler\nmean=0 · std=1]
    I --> J[Split Train/Test\ntrain_test_split\ntest_size=0.31 · stratify=y]
    J --> K([Selesai Preprocessing])
```

---

## 3. Training Logistic Regression

```mermaid
flowchart TD
    A([Mulai LR]) --> B[Import\nLogisticRegression\nfrom sklearn.linear_model]
    B --> C[Inisialisasi Model\nLogisticRegression\nmax_iter=1000\nmulti_class=auto]
    C --> D[Fit Model\nmodel.fit\nX_train · y_train]
    D --> E[Prediksi\nmodel.predict\nX_test]
    E --> F[Hitung Metrik\naccuracy_score\nf1_score\nclassification_report]
    F --> G{Akurasi\n≥ 90%?}
    G -- Tidak --> H[Tuning Hyperparameter\nC · solver · penalty]
    H --> C
    G -- Ya --> I[Simpan Hasil\nClassification Report]
    I --> J([Selesai LR\nAcc: 0.9031])
```

---

## 4. Training Support Vector Machine (SVM)

```mermaid
flowchart TD
    A([Mulai SVM]) --> B[Import\nSVC\nfrom sklearn.svm]
    B --> C[Inisialisasi Model\nSVC\nkernel=rbf\nC=1.0 · gamma=scale]
    C --> D[Fit Model\nsvm.fit\nX_train · y_train]
    D --> E[Prediksi\nsvm.predict\nX_test]
    E --> F[Hitung Metrik\naccuracy_score\nf1_score\nclassification_report]
    F --> G{Akurasi\n≥ 90%?}
    G -- Tidak --> H[Tuning Hyperparameter\nGridSearchCV\nC · gamma · kernel]
    H --> C
    G -- Ya --> I[Simpan Hasil\nClassification Report]
    I --> J([Selesai SVM\nAcc: 0.9128])
```

---

## 5. Training Random Forest

```mermaid
flowchart TD
    A([Mulai RF]) --> B[Import\nRandomForestClassifier\nfrom sklearn.ensemble]
    B --> C[Inisialisasi Model\nRandomForestClassifier\nn_estimators=200\nrandom_state=42]
    C --> D[Fit Model\nrf.fit\nX_train · y_train]
    D --> E[Prediksi\nrf.predict\nX_test]
    E --> F[Hitung Metrik\naccuracy_score\nf1_score\nclassification_report]
    F --> G[Plot Feature Importance\nsuhu · kelembaban · kadar_gas]
    G --> H{Akurasi\n≥ 90%?}
    H -- Tidak --> I[Tuning Hyperparameter\nmax_depth · min_samples_split\nmax_features]
    I --> C
    H -- Ya --> J[Simpan Hasil\nClassification Report\n+ Feature Importance]
    J --> K([Selesai RF\nAcc: 0.9370])
```

---

## 6. Training RNN / LSTM

```mermaid
flowchart TD
    A([Mulai LSTM]) --> B[Import TensorFlow/Keras\nLSTM · Bidirectional\nDense · Dropout]
    B --> C[Reshape Data ke Sequence\nWindow = 10 jam\nformat: samples · timesteps · features\n= N · 10 · 3]
    C --> D[Split Train/Test\nSequence-aware split]
    D --> E[Bangun Arsitektur Model\nBidirectional LSTM 64 unit\nDropout 0.3\nBidirectional LSTM 32 unit\nDropout 0.2\nDense 3 unit · Softmax]
    E --> F[Kompilasi Model\noptimizer=adam\nloss=sparse_categorical_crossentropy\nmetrics=accuracy]
    F --> G[Training\nmodel.fit\nX_train · y_train\nepochs=50\nbatch_size=32\nvalidation_split=0.2]
    G --> H[Plot Training History\nLoss & Accuracy Curve\nTrain vs Validation]
    H --> I{Overfitting\natau\nUnderfitting?}
    I -- Overfitting --> J[Tambah Dropout\natau Kurangi Epoch]
    I -- Underfitting --> K[Tambah Layer\natau Tingkatkan Epoch]
    J --> F
    K --> F
    I -- Tidak --> L[Evaluasi\nmodel.evaluate\nX_test · y_test]
    L --> M[Prediksi\nnp.argmax\nmodel.predict · X_test]
    M --> N[Hitung Metrik\naccuracy_score\nclassification_report]
    N --> O[Simpan Model\nmodel.save]
    O --> P([Selesai LSTM\nAcc: 0.9646])
```

---

## 7. Evaluasi & Perbandingan Model

```mermaid
flowchart TD
    A([Mulai Evaluasi]) --> B[Kumpulkan Hasil\nSemua 4 Model]
    B --> C[Buat Tabel Perbandingan\nAccuracy · F1 Score\nPrecision · Recall]
    C --> D[Plot Confusion Matrix\nSetiap Model]
    D --> E[Plot Bar Chart\nPerbandingan Accuracy]
    E --> F{Model\nTerbaik?}
    F --> G[LSTM\nAcc 0.9646\nTime Series Aware]
    F --> H[Random Forest\nAcc 0.9370\nInterpretable]
    F --> I[SVM\nAcc 0.9128\nRobust Outlier]
    F --> J[Logistic Reg\nAcc 0.9031\nPaling Cepat]
    G --> K{Kriteria\nPemilihan}
    H --> K
    I --> K
    J --> K
    K -- Akurasi Tertinggi --> L[✅ Pilih LSTM]
    K -- Interpretabilitas --> M[✅ Pilih Random Forest]
    K -- Kecepatan Training --> N[✅ Pilih Logistic Regression]
    L --> O([Selesai Evaluasi])
    M --> O
    N --> O
```

---

## 8. Ringkasan Alur Per Model

| Tahap | Logistic Regression | SVM | Random Forest | LSTM |
|-------|-------------------|-----|---------------|------|
| **Input** | `[N, 3]` flat | `[N, 3]` flat | `[N, 3]` flat | `[N, 10, 3]` sequence |
| **Normalisasi** | StandardScaler | StandardScaler | Tidak wajib | MinMaxScaler |
| **Training** | `model.fit()` | `svm.fit()` | `rf.fit()` | `model.fit()` epochs |
| **Tuning** | C, solver | C, gamma, kernel | n_estimators, depth | epochs, dropout, units |
| **Output** | Koefisien linear | Support vectors | Decision trees + feature importance | Bobot neural network |
| **Akurasi** | 0.9031 | 0.9128 | 0.9370 | **0.9646** |
