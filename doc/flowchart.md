# Training Implementation Flowchart — Cassava Tapai Ripeness Prediction

The flow of this research is carried out as illustrated in the flowchart below. There are 4 models that will be used in this research, namely Logistic Regression, Support Vector Machine (SVM), Random Forest, and LSTM (Long Short-Term Memory). These four models were selected because they represent different architectural characteristics in terms of linearity, kernel-based transformation, ensemble learning, and temporal sequence modeling. At the same time, the selection allows a comprehensive comparison between classical machine learning approaches and deep learning. After sharing the dataset, all models will be subjected to the same preprocessing and trained with various parameters. Then all models will be tested and evaluated, so that the model with the best accuracy can be determined.

---

## 1. General Overview

```mermaid
flowchart TD
    A([🟢 Start]) --> B[Load Dataset\ndataset_kematangan_tapai.csv]
    B --> C[Exploratory Data Analysis\nDescriptive Statistics]
    C --> D{Data\nClean?}
    D -- No --> E[Preprocessing\nHandle Missing Values\nRemove Duplicates]
    E --> F[Feature Engineering\ntemperature · humidity · gas_level]
    D -- Yes --> F
    F --> G[Encode Labels\nunripe · ripe · overripe]
    G --> H[Split Data\nTrain 69% · Test 31%]
    H --> I{Select\nModel}
    I --> J[Logistic\nRegression]
    I --> K[Support Vector\nMachine]
    I --> L[Random\nForest]
    I --> M[RNN / LSTM]
    J --> N[Evaluate Model]
    K --> N
    L --> N
    M --> N
    N --> O[Compare\nAccuracy & F1 Score]
    O --> P([🔴 End])
```

---

## 2. Preprocessing & Data Preparation

```mermaid
flowchart TD
    A([Start Preprocessing]) --> B[Read CSV\npd.read_csv]
    B --> C[Check Dataset Info]
    C --> D[Check Missing Values]
    D --> E{Missing\nValues\nExist?}
    E -- Yes --> F[Fill or Drop\nMissing Values]
    E -- No --> G[Select Features\ntemperature · humidity · gas_level]
    F --> G
    G --> H[Encode Target Label\nLabelEncoder\nunripe=0 · ripe=1 · overripe=2]
    H --> I[Normalize Features\nStandardScaler\nmean=0 · std=1]
    I --> J[Train/Test Split\ntrain_test_split\ntest_size=0.31 · stratify=y]
    J --> K([Preprocessing Done])
```

---

## 3. Logistic Regression Training

```mermaid
flowchart TD
    A([Start LR]) --> B[Import\nLogisticRegression\nfrom sklearn.linear_model]
    B --> C[Initialize Model\nLogisticRegression\nmax_iter=1000\nmulti_class=auto]
    C --> D[Fit Model\nmodel.fit\nX_train · y_train]
    D --> E[Predict\nmodel.predict\nX_test]
    E --> F[Compute Metrics\naccuracy_score\nf1_score\nclassification_report]
    F --> G{Accuracy\n≥ 90%?}
    G -- No --> H[Tune Hyperparameters\nC · solver · penalty]
    H --> C
    G -- Yes --> I[Save Results\nClassification Report]
    I --> J([LR Done\nAcc: 0.9031])
```

---

## 4. Support Vector Machine (SVM) Training

```mermaid
flowchart TD
    A([Start SVM]) --> B[Import\nSVC\nfrom sklearn.svm]
    B --> C[Initialize Model\nSVC\nkernel=rbf\nC=1.0 · gamma=scale]
    C --> D[Fit Model\nsvm.fit\nX_train · y_train]
    D --> E[Predict\nsvm.predict\nX_test]
    E --> F[Compute Metrics\naccuracy_score\nf1_score\nclassification_report]
    F --> G{Accuracy\n≥ 90%?}
    G -- No --> H[Tune Hyperparameters\nGridSearchCV\nC · gamma · kernel]
    H --> C
    G -- Yes --> I[Save Results\nClassification Report]
    I --> J([SVM Done\nAcc: 0.9128])
```

---

## 5. Random Forest Training

```mermaid
flowchart TD
    A([Start RF]) --> B[Import\nRandomForestClassifier\nfrom sklearn.ensemble]
    B --> C[Initialize Model\nRandomForestClassifier\nn_estimators=200\nrandom_state=42]
    C --> D[Fit Model\nrf.fit\nX_train · y_train]
    D --> E[Predict\nrf.predict\nX_test]
    E --> F[Compute Metrics\naccuracy_score\nf1_score\nclassification_report]
    F --> G[Plot Feature Importance\ntemperature · humidity · gas_level]
    G --> H{Accuracy\n≥ 90%?}
    H -- No --> I[Tune Hyperparameters\nmax_depth · min_samples_split\nmax_features]
    I --> C
    H -- Yes --> J[Save Results\nClassification Report\n+ Feature Importance]
    J --> K([RF Done\nAcc: 0.9370])
```

---

## 6. RNN / LSTM Training

```mermaid
flowchart TD
    A([Start LSTM]) --> B[Import TensorFlow/Keras\nLSTM · Bidirectional\nDense · Dropout]
    B --> C[Reshape Data into Sequences\nWindow = 10 hours\nshape: samples · timesteps · features\n= N · 10 · 3]
    C --> D[Train/Test Split\nSequence-aware split]
    D --> E[Build Model Architecture\nBidirectional LSTM 64 units\nDropout 0.3\nBidirectional LSTM 32 units\nDropout 0.2\nDense 3 units · Softmax]
    E --> F[Compile Model\noptimizer=adam\nloss=sparse_categorical_crossentropy\nmetrics=accuracy]
    F --> G[Train\nmodel.fit\nX_train · y_train\nepochs=50\nbatch_size=32\nvalidation_split=0.2]
    G --> H[Plot Training History\nLoss & Accuracy Curve\nTrain vs Validation]
    H --> I{Overfitting\nor\nUnderfitting?}
    I -- Overfitting --> J[Increase Dropout\nor Reduce Epochs]
    I -- Underfitting --> K[Add More Layers\nor Increase Epochs]
    J --> F
    K --> F
    I -- Neither --> L[Evaluate\nmodel.evaluate\nX_test · y_test]
    L --> M[Predict\nnp.argmax\nmodel.predict · X_test]
    M --> N[Compute Metrics\naccuracy_score\nclassification_report]
    N --> O[Save Model\nmodel.save]
    O --> P([LSTM Done\nAcc: 0.9646])
```

---

## 7. Model Evaluation & Comparison

```mermaid
flowchart TD
    A([Start Evaluation]) --> B[Collect Results\nAll 4 Models]
    B --> C[Build Comparison Table\nAccuracy · F1 Score\nPrecision · Recall]
    C --> D[Plot Confusion Matrix\nPer Model]
    D --> E[Plot Bar Chart\nAccuracy Comparison]
    E --> F{Best\nModel?}
    F --> G[LSTM\nAcc 0.9646\nTime Series Aware]
    F --> H[Random Forest\nAcc 0.9370\nInterpretable]
    F --> I[SVM\nAcc 0.9128\nRobust to Outliers]
    F --> J[Logistic Reg\nAcc 0.9031\nFastest Training]
    G --> K{Selection\nCriteria}
    H --> K
    I --> K
    J --> K
    K -- Highest Accuracy --> L[✅ Choose LSTM]
    K -- Interpretability --> M[✅ Choose Random Forest]
    K -- Training Speed --> N[✅ Choose Logistic Regression]
    L --> O([Evaluation Done])
    M --> O
    N --> O
```

---

## 8. Per-Model Summary

| Stage             | Logistic Regression | SVM              | Random Forest                       | LSTM                   |
| ----------------- | ------------------- | ---------------- | ----------------------------------- | ---------------------- |
| **Input**         | `[N, 3]` flat       | `[N, 3]` flat    | `[N, 3]` flat                       | `[N, 10, 3]` sequence  |
| **Normalization** | StandardScaler      | StandardScaler   | Not required                        | MinMaxScaler           |
| **Training**      | `model.fit()`       | `svm.fit()`      | `rf.fit()`                          | `model.fit()` epochs   |
| **Tuning**        | C, solver           | C, gamma, kernel | n_estimators, depth                 | epochs, dropout, units |
| **Output**        | Linear coefficients | Support vectors  | Decision trees + feature importance | Neural network weights |
| **Accuracy**      | 0.9031              | 0.9128           | 0.9370                              | **0.9646**             |
