## Classification Methods

This study applies four machine learning classification methods to predict the ripeness status of tapai — a traditional fermented cassava food — based on sensor data collected during the fermentation process. The dataset consists of 600 observations drawn from 10 independent fermentation trials, each spanning 60 hours of continuous monitoring. Three input features are used across all models: hour of observation (_jam_), relative humidity (_kelembaban_), and gas concentration (_kadar_gas_). The target variable is a three-class label representing ripeness status: _unripe_ (belum matang), _ripe_ (matang), and _overripe_ (terlalu matang).

---

### 3.1 Logistic Regression

The first classification approach employs Logistic Regression, a linear discriminant model that estimates class membership probabilities through a softmax activation function applied to a linear combination of input features. Prior to model fitting, the input features are standardized to zero mean and unit variance, and the target labels are numerically encoded to represent each ripeness category. The dataset is partitioned into training and testing subsets using a stratified random split, retaining 31% of observations for evaluation. Each hourly observation is treated as an independent sample; no temporal dependencies between consecutive time steps are modeled. The model learns a separate weight vector for each class, and the decision boundary is defined as a linear hyperplane in the three-dimensional feature space.

The classification report of this model is as follows:

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Unripe       | 0.96      | 0.92   | 0.94     | 136     |
| Ripe         | 0.82      | 0.88   | 0.85     | 130     |
| Overripe     | 0.93      | 0.90   | 0.92     | 147     |
| **Accuracy** |           |        | **0.90** | **413** |
| Macro avg    | 0.90      | 0.90   | 0.90     | 413     |
| Weighted avg | 0.91      | 0.90   | 0.90     | 413     |

The model achieves an overall accuracy of 0.9031 and a macro-averaged F1-score of 0.90. The relatively lower precision for the _ripe_ class (0.82) suggests that the linear decision boundary struggles to cleanly separate this intermediate class from its adjacent categories.

---

### 3.2 Support Vector Machine

The second approach applies a Support Vector Machine (SVM) classifier using the Radial Basis Function (RBF) kernel. Unlike Logistic Regression, the RBF kernel implicitly maps the input features into a higher-dimensional space, enabling the model to learn non-linear decision boundaries. The SVM optimization objective seeks the hyperplane that maximizes the margin between the nearest support vectors of each class pair. The same data preparation procedure used in Section 3.1 — feature standardization and label encoding — is applied prior to training. Each hourly observation continues to be treated as an independent data point, with no sequential context incorporated.

The classification report of this model is as follows:

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Unripe       | 0.95      | 0.95   | 0.95     | 136     |
| Ripe         | 0.85      | 0.88   | 0.86     | 130     |
| Overripe     | 0.94      | 0.90   | 0.92     | 147     |
| **Accuracy** |           |        | **0.91** | **413** |
| Macro avg    | 0.91      | 0.91   | 0.91     | 413     |
| Weighted avg | 0.91      | 0.91   | 0.91     | 413     |

The SVM achieves an accuracy of 0.9128 and a macro-averaged F1-score of 0.91, representing a modest improvement over Logistic Regression. The RBF kernel's capacity to capture non-linear relationships in the feature space contributes to better discrimination of the _ripe_ class, as evidenced by the increase in precision from 0.82 to 0.85 relative to the linear baseline.

---

### 3.3 Random Forest

The third method employs a Random Forest classifier, an ensemble learning algorithm that constructs a large number of decision trees in parallel through bootstrap aggregating and random feature subsampling. In this implementation, 200 individual decision trees are trained on randomly drawn subsets of the training data, each considering a random subset of features at each split node. The final class prediction for each sample is determined by majority vote across all trees. Unlike the previous models, Random Forest does not require explicit feature scaling, as decision tree splits are invariant to monotonic transformations of the input features. The same stratified train-test split is retained, and the target labels are numerically encoded prior to training.

The classification report of this model is as follows:

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Unripe       | 0.98      | 0.95   | 0.96     | 136     |
| Ripe         | 0.87      | 0.94   | 0.90     | 130     |
| Overripe     | 0.96      | 0.93   | 0.94     | 147     |
| **Accuracy** |           |        | **0.94** | **413** |
| Macro avg    | 0.94      | 0.94   | 0.94     | 413     |
| Weighted avg | 0.94      | 0.94   | 0.94     | 413     |

Random Forest achieves the highest accuracy among the three non-temporal models at 0.9370, with a macro-averaged F1-score of 0.94. The ensemble mechanism substantially reduces variance compared to a single decision tree, and the model's inherent non-linearity allows it to capture interaction effects between humidity, gas concentration, and observation hour. Additionally, feature importance scores are available as a by-product of training, providing interpretable insight into predictor contributions.

---

### 3.4 Long Short-Term Memory (LSTM) Recurrent Neural Network

The fourth approach treats the fermentation data as a genuine time series and applies a Bidirectional Long Short-Term Memory (LSTM) network, a specialized variant of Recurrent Neural Networks (RNN) designed to model sequential dependencies in temporal data (Hochreiter & Schmidhuber, 1997). Rather than treating each hourly observation as an independent sample, the dataset is restructured into overlapping sliding windows of 10 consecutive time steps, so that each training sample represents a contiguous sequence of 10 hours of sensor readings across three measured variables. The Bidirectional LSTM architecture processes each sequence in both forward and backward temporal directions, allowing the model to exploit contextual information from both past and future observations within each window. The network is trained by minimizing a multi-class classification loss function through an iterative gradient-based optimization procedure, and a stratified split is applied at the trial level to prevent data leakage across fermentation experiments. The model is trained over 50 iterations with a batch size of 32 samples per update.

The network architecture consists of the following sequential layers:

- **First LSTM Layer (64 units)** — Processes the input sequence and learns temporal patterns across the 3-hour observation window, passing its intermediate outputs forward to the next layer.
- **First Dropout Layer (rate = 0.2)** — Randomly deactivates 20% of neurons during training to reduce overfitting and improve generalization.
- **Second LSTM Layer (32 units)** — Further distills the temporal representations learned by the first layer into a compact feature vector summarizing the entire sequence.
- **Second Dropout Layer (rate = 0.2)** — Applies the same regularization mechanism as the first dropout layer to the condensed representations.
- **Output Layer (3 units)** — A fully connected layer that maps the final learned representation to probability scores for each of the three ripeness classes, using a softmax function to ensure the probabilities sum to one.

The classification report of this model is as follows:

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Unripe       | 1.00      | 0.99   | 0.99     | 138     |
| Ripe         | 0.83      | 0.86   | 0.84     | 22      |
| Overripe     | 0.92      | 0.95   | 0.94     | 38      |
| **Accuracy** |           |        | **0.96** | **198** |
| Macro avg    | 0.92      | 0.93   | 0.92     | 198     |
| Weighted avg | 0.97      | 0.96   | 0.97     | 198     |

The LSTM model achieves the highest overall accuracy of 0.9646 across all four methods. The near-perfect precision of 1.00 for the _unripe_ class reflects the model's ability to reliably identify the early fermentation phase by leveraging the cumulative temporal pattern of rising gas concentration and humidity. The macro-averaged F1-score of 0.92 — computed on a smaller test set with class imbalance due to the sliding window construction procedure — reflects the model's strong generalization to unseen fermentation trials. The explicit modeling of temporal dependencies renders this approach the most scientifically appropriate for the fermentation domain, where the ripeness status at any given hour is inherently conditioned on the physiochemical trajectory of preceding hours.
