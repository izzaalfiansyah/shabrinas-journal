Tapai singkong (cassava tapai) is a traditional Indonesian fermented food product whose
quality is highly dependent on precise control of the fermentation process. Inconsistent
fermentation outcomes arise from fluctuating environmental conditions — including temperature,
humidity, and fermentation gas levels — making it difficult to reliably determine ripeness status
without objective measurement tools. This study addresses the challenge of automated ripeness
prediction by providing a controlled, head-to-head comparison of four machine learning approaches
— Logistic Regression, Support Vector Machine (SVM), Random Forest, and LSTM-based
Recurrent Neural Network (RNN) — on a single, uniformly preprocessed dataset of 600
time-series observations across three ripeness classes (belum matang, matang, terlalu matang),
collected from 10 fermentation trials spanning 60 hours each. All models were evaluated under
identical preprocessing and hyperparameter settings using accuracy, precision, recall, F1-score,
and confusion matrices to reveal per-class behavior. LSTM yielded the best test performance
(96.46% accuracy; macro F1 = 0.93), Random Forest followed closely (93.70% accuracy; macro
F1 = 0.94), while SVM and Logistic Regression obtained 91.28% and 90.31% accuracy
respectively. This paper discusses the trade-off between predictive performance, temporal
modeling capability, and interpretability, and recommends LSTM for high-accuracy quality
control deployments where temporal dependencies are critical, and Random Forest as a strong,
interpretable alternative for resource-constrained environments. Per-class metrics and experimental
artifacts are provided to support reproducibility and practical adoption in traditional food
production monitoring.

