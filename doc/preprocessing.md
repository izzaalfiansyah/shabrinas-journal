# Dataset Preprocessing

One of the most important stages in machine learning development is dataset preprocessing. Before training, the dataset must be preprocessed first to ensure compatibility with the models used. It is important to note that the cleaned data produced by this preprocessing pipeline is used exclusively for classical machine learning models (Logistic Regression, SVM, and Random Forest). The RNN model, on the other hand, uses the original raw dataset, as it requires data with temporal sequence interpretation — specifically, the time-series ordering of sensor readings across each fermentation trial.

## Raw Dataset Overview

The raw dataset (`dataset_kematangan_tapai.csv`) consists of **600 records** and **6 columns**: `percobaan` (trial number), `jam` (hour), `suhu` (temperature), `kelembaban` (humidity), `kadar_gas` (gas level), and `status_kematangan` (ripeness status). The target class distribution is heavily imbalanced:

| Class          | Count |
| -------------- | ----- |
| belum matang   | 417   |
| terlalu matang | 116   |
| matang         | 67    |

This imbalance can cause classical models to be biased toward the majority class, which is addressed in the preprocessing step.

## Preprocessing Steps

### 1. Missing Value Handling

The first step is to check and remove any missing values from the dataset. This ensures that no incomplete records are passed to the model, which could lead to errors or degraded performance.

```python
df = pd.read_csv('../dataset/dataset_kematangan_tapai.csv')
df.dropna(inplace=True)
```

In this dataset, no missing values were found, so all 600 records are retained after this step.

### 2. Class Balancing with SMOTE

Due to the significant imbalance in class distribution, the Synthetic Minority Over-sampling Technique (SMOTE) is applied. SMOTE generates synthetic data points for the minority classes by interpolating between existing samples in the feature space, rather than simply duplicating existing records.

The feature columns used for resampling are: `suhu`, `kelembaban`, `kadar_gas`, and `jam`. The column `percobaan` is excluded as it is an identifier, not a sensor feature.

```python
from imblearn.over_sampling import SMOTE

x = df[['suhu', 'kelembaban', 'kadar_gas', 'jam']]
y = df['status_kematangan']

sample = SMOTE()
transformed_x, transformed_y = sample.fit_resample(x, y)
```

After applying SMOTE, each class contains **417 records**, resulting in a balanced dataset of **1,251 records** in total:

| Class          | Before SMOTE | After SMOTE |
| -------------- | ------------ | ----------- |
| belum matang   | 417          | 417         |
| matang         | 67           | 417         |
| terlalu matang | 116          | 417         |

The balanced dataset is then saved as the cleaned dataset:

```python
transformed_df = pd.concat([transformed_x, transformed_y], axis=1)
transformed_df.to_csv('../dataset/dataset_kematangan_tapai_cleaned.csv')
```

This balancing step aims to ensure that the developed classical models are able to learn features from all classes equally and improve generalization on previously unrecognized data patterns.

### 3. Feature Selection

For classical model training, only the three most relevant sensor features are selected as input columns: `jam` (hour), `kelembaban` (humidity), and `kadar_gas` (gas level). The `suhu` (temperature) feature is excluded from model input based on feature analysis. The output column is `status_kematangan`.

```python
input_columns = ['jam', 'kelembaban', 'kadar_gas']
output_columns = ['status_kematangan']
```

### 4. Label Encoding

Since the target variable `status_kematangan` is a categorical string label, it must be converted into a numerical representation for model compatibility. A `LabelEncoder` is used to assign integer codes to each class.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['status_kematangan'] = label_encoder.fit_transform(data['status_kematangan'])
```

### 5. Feature Scaling

All input features are standardized using `StandardScaler`, which transforms each feature to have a mean of zero and a standard deviation of one. This normalization step is particularly important for distance-based and gradient-based models (such as SVM and Logistic Regression) to prevent features with larger magnitudes from dominating the learning process.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(data[input_columns].values)
```

### 6. Train-Test Split

The preprocessed data is split into training and testing sets with a ratio of **67% training and 33% testing**, using a fixed random state for reproducibility.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
```

## Summary

The full preprocessing pipeline for classical models consists of the following stages:

1. **Missing value removal** — ensures data completeness
2. **SMOTE oversampling** — resolves class imbalance
3. **Feature selection** — retains only `jam`, `kelembaban`, and `kadar_gas`
4. **Label encoding** — converts categorical labels to integers
5. **Feature scaling** — standardizes input features with `StandardScaler`
6. **Train-test split** — partitions data into 67% train and 33% test

> **Note:** The RNN model does not use the cleaned dataset. It operates on the original raw dataset (`dataset_kematangan_tapai.csv`) directly, preserving the temporal ordering of sensor readings grouped by `percobaan` (trial), which is essential for constructing time-series sequences used in LSTM-based sequence modeling.
