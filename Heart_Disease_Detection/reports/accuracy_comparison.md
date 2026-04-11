# Accuracy Comparison Report

## Decision Tree Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.6721 |
| Precision | 0.7097 |
| Recall | 0.6667 |
| F1-score | 0.6875 |

Best hyperparameters: `{'classifier__criterion': 'entropy', 'classifier__max_depth': 5, 'classifier__min_samples_leaf': 4, 'classifier__min_samples_split': 10}`

## Expert System Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.2459 |
| Precision | 0.2593 |
| Recall | 0.2121 |
| F1-score | 0.2333 |

## Observations

- The decision tree is tuned with grid search and typically achieves stronger predictive performance on the validation set.
- The expert system is more transparent because every prediction is tied to explicit rules.
- Combining both approaches in the Streamlit app gives users both a data-driven score and a human-readable explanation.
