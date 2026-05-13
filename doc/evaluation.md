## Model Evaluation

This section describes the evaluation methodology applied to assess the predictive performance of all four classification models examined in this study. The evaluation framework is grounded in standard information retrieval and binary classification theory, extended to the multi-class setting through macro and weighted averaging strategies. For each model, four primary metrics are computed: Precision, Recall, F1-Score, and Accuracy. These metrics are derived from the confusion matrix, which summarizes the agreement between predicted and true class labels on the held-out test set.

---

### 4.1 Confusion Matrix

The foundation of all classification metrics used in this study is the **confusion matrix**, a square matrix of dimension $K \times K$, where $K$ denotes the number of target classes. In this study, $K = 3$, corresponding to the three ripeness categories: _unripe_ (belum matang), _ripe_ (matang), and _overripe_ (terlalu matang). Each entry $C_{ij}$ of the confusion matrix represents the number of observations belonging to true class $i$ that the model predicted as class $j$.

For a given class $k$, four fundamental counts are derived from the confusion matrix:

- **True Positives (TP$_k$)**: The number of instances correctly predicted as class $k$ (i.e., the diagonal entry $C_{kk}$).
- **False Positives (FP$_k$)**: The number of instances from other classes incorrectly predicted as class $k$ (i.e., the sum of column $k$ excluding the diagonal: $\sum_{i \neq k} C_{ik}$).
- **False Negatives (FN$_k$)**: The number of instances of class $k$ incorrectly predicted as another class (i.e., the sum of row $k$ excluding the diagonal: $\sum_{j \neq k} C_{kj}$).
- **True Negatives (TN$_k$)**: The number of instances correctly identified as not belonging to class $k$ (i.e., the total number of instances minus $\text{TP}_k$, $\text{FP}_k$, and $\text{FN}_k$).

These per-class counts form the basis for computing the four evaluation metrics described in the following subsections.

---

### 4.2 Precision

**Precision** measures the proportion of predicted positive instances for a given class that are truly positive. It quantifies the classifier's ability to avoid false positives — that is, the degree to which positive predictions are reliable. For class $k$, Precision is formally defined as:

$$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$$

A high Precision value for class $k$ indicates that, when the model predicts an instance as belonging to class $k$, it is correct the majority of the time. Conversely, a low Precision value signals that the model frequently misclassifies instances from other classes as class $k$. In the context of ripeness classification, high Precision for the _ripe_ class, for instance, implies that samples predicted as ripe are very likely to be genuinely ripe, which is critical for practical quality assurance applications.

When aggregated across all classes, Precision is reported as a **macro-averaged** score, computed as the unweighted arithmetic mean of per-class Precision values:

$$\text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}_k$$

A **weighted average** variant is also reported, in which the contribution of each class is proportional to its support (the number of true instances in the test set):

$$\text{Precision}_{\text{weighted}} = \frac{\sum_{k=1}^{K} \text{support}_k \cdot \text{Precision}_k}{\sum_{k=1}^{K} \text{support}_k}$$

---

### 4.3 Recall

**Recall** — also referred to as Sensitivity or True Positive Rate — measures the proportion of actual positive instances for a given class that are correctly identified by the model. It quantifies the classifier's ability to avoid false negatives. For class $k$, Recall is defined as:

$$\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$

A high Recall value for class $k$ indicates that the model successfully identifies most true instances of that class. A low Recall value, conversely, suggests that the model frequently fails to recognize instances that genuinely belong to class $k$, instead misclassifying them into other categories. In the domain of fermentation monitoring, high Recall for the _overripe_ class is particularly important, as failing to detect overripe samples could result in degraded product quality reaching end consumers.

Macro-averaged and weighted-averaged Recall are computed analogously to the Precision aggregation described in Section 4.2:

$$\text{Recall}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Recall}_k$$

$$\text{Recall}_{\text{weighted}} = \frac{\sum_{k=1}^{K} \text{support}_k \cdot \text{Recall}_k}{\sum_{k=1}^{K} \text{support}_k}$$

---

### 4.4 F1-Score

The **F1-Score** is the harmonic mean of Precision and Recall, providing a single metric that balances both concerns. It is particularly informative in scenarios where one metric alone could be misleading — for instance, a trivial classifier that predicts all instances as the majority class would achieve high Recall for that class while incurring severe Precision penalties. The F1-Score for class $k$ is defined as:

$$\text{F1}_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

The harmonic mean penalizes extreme imbalances between Precision and Recall more severely than the arithmetic mean, thereby rewarding models that achieve simultaneous improvements in both metrics. An F1-Score approaching 1.0 indicates near-perfect discrimination of class $k$, while a value approaching 0 indicates poor classification performance for that class.

Macro-averaged and weighted-averaged F1-Scores are computed following the same aggregation procedure:

$$\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k$$

$$\text{F1}_{\text{weighted}} = \frac{\sum_{k=1}^{K} \text{support}_k \cdot \text{F1}_k}{\sum_{k=1}^{K} \text{support}_k}$$

In multi-class settings with imbalanced class distributions, the **macro-averaged F1-Score** is particularly informative because it assigns equal weight to each class regardless of its frequency, thereby highlighting any systematic deficiency in classifying minority classes. The **weighted-averaged F1-Score**, by contrast, reflects the overall expected performance across the observed class distribution in the test set.

---

### 4.5 Accuracy

**Accuracy** is the most straightforward evaluation metric, defined as the proportion of all test instances that the model correctly classifies. Across all $N$ test samples and $K$ classes, Accuracy is computed as:

$$\text{Accuracy} = \frac{\sum_{k=1}^{K} \text{TP}_k}{N} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$$

While Accuracy provides an intuitive overall measure of classification performance, it can be misleading when class distributions are imbalanced, as a model biased toward the majority class may appear to perform well by this metric alone. In this study, the training and test sets are constructed using stratified sampling to preserve the approximate class distribution of the original dataset, thereby mitigating the risk of inflated Accuracy estimates. Nonetheless, Accuracy is reported alongside per-class Precision, Recall, and F1-Score to provide a comprehensive characterization of model behavior.

---

### 4.6 Evaluation Procedure

All evaluation metrics are computed on the held-out test partition, which the model has not encountered during training or hyperparameter selection. The test partition is constructed using a stratified random split, which ensures that the proportional representation of each ripeness class is preserved across both the training and test subsets. For the three non-temporal models (Logistic Regression, SVM, and Random Forest), 31% of the total dataset is reserved for evaluation, yielding a test set of 413 observations drawn from the full 600-observation pool. For the LSTM model, partitioning is performed at the fermentation trial level — entire trials are assigned to either the training or test set — to prevent data leakage arising from the temporal continuity between consecutive hourly observations within the same fermentation run. This yields a test set of 198 windowed sequences derived from the held-out trials.

The `classification_report` function from the scikit-learn library is used to compute per-class Precision, Recall, and F1-Score for the three non-temporal models, while the same metrics are derived from the confusion matrix output of TensorFlow/Keras model predictions for the LSTM. Accuracy is computed using the `accuracy_score` function (scikit-learn) and the `evaluate` method (Keras), respectively.

The per-class metrics are subsequently aggregated into macro-averaged and weighted-averaged summary statistics for inter-model comparison. The complete set of evaluation results — including per-class breakdowns and summary averages — is presented in Section 3 for each individual model, and consolidated into a comparative summary table in the subsequent section of this paper.

---

### 4.7 Summary of Evaluation Results

The following table presents a consolidated comparison of all four classification models based on the macro-averaged Precision, Recall, F1-Score, and overall Accuracy computed on their respective test sets.

| Model                    | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Accuracy |
| ------------------------ | :---------------: | :------------: | :--------------: | :------: |
| Logistic Regression      | 0.90              | 0.90           | 0.90             | 0.9031   |
| Support Vector Machine   | 0.91              | 0.91           | 0.91             | 0.9128   |
| Random Forest            | 0.94              | 0.94           | 0.94             | 0.9370   |
| LSTM (Bidirectional RNN) | 0.92              | 0.93           | 0.92             | 0.9646   |

Among the four models, the LSTM achieves the highest overall Accuracy (0.9646), reflecting the advantage of explicitly modeling temporal dependencies in the fermentation time series. Random Forest achieves the highest macro-averaged F1-Score (0.94) among the non-temporal models, demonstrating that ensemble-based non-linear classifiers are substantially more effective than linear approaches for this classification task. The SVM improves upon Logistic Regression across all metrics by leveraging the RBF kernel to capture non-linear decision boundaries. Logistic Regression, as the simplest baseline, still achieves competitive performance with an Accuracy of 0.9031, confirming that the three sensor features carry strong discriminative information even when temporal context is disregarded.

The lower macro-averaged F1-Score of the LSTM (0.92) relative to its high Accuracy (0.9646) reflects the influence of class imbalance in the LSTM test set — specifically, the disproportionately small support for the _ripe_ class (22 instances) after sliding-window construction and trial-level partitioning. The macro average penalizes this underperformance equally regardless of class frequency, whereas the weighted-average F1-Score of 0.97 is more representative of the LSTM's practical utility across the observed class distribution.