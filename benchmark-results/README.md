# Encoding benchmark results

Dataset SHA-256: `2a9f028bd612f90ce7bb94dc798bd5748d6d1811ad4d740aad5c25187a4459a3`

Primary selection metric: nested-CV outer-fold macro F1. Lower Brier/ECE is better.

| Model | Encoding | Accuracy | Balanced accuracy | Macro F1 | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | one-hot | 0.9676 ± 0.0101 | 0.9175 ± 0.0562 | 0.9244 ± 0.0494 | 0.0648 | 0.0324 |
| Decision Tree | ordinal | 0.9740 ± 0.0149 | 0.9404 ± 0.0196 | 0.9423 ± 0.0270 | 0.0485 | 0.0247 |
| Gradient Boosting | one-hot | 0.9861 ± 0.0056 | 0.9848 ± 0.0164 | 0.9723 ± 0.0202 | 0.0342 | 0.0476 |
| Gradient Boosting | ordinal | 0.9821 ± 0.0038 | 0.9759 ± 0.0191 | 0.9644 ± 0.0162 | 0.0409 | 0.0560 |
| K-Nearest Neighbors | one-hot | 0.9184 ± 0.0171 | 0.7514 ± 0.0271 | 0.8065 ± 0.0167 | 0.1719 | 0.1496 |
| K-Nearest Neighbors | ordinal | 0.9254 ± 0.0313 | 0.7897 ± 0.0756 | 0.8291 ± 0.0703 | 0.1627 | 0.0950 |
| Logistic Regression | one-hot | 0.9334 ± 0.0058 | 0.8892 ± 0.0416 | 0.8863 ± 0.0252 | 0.0902 | 0.0145 |
| Logistic Regression | ordinal | 0.6886 ± 0.0158 | 0.3270 ± 0.0244 | 0.3257 ± 0.0212 | 0.3959 | 0.0697 |
| Naive Bayes | one-hot | 0.8027 ± 0.0184 | 0.8620 ± 0.0364 | 0.7394 ± 0.0378 | 0.3946 | 0.1973 |
| Naive Bayes | ordinal | 0.6719 ± 0.0218 | 0.4947 ± 0.0114 | 0.3294 ± 0.0165 | 0.5248 | 0.1751 |
| Random Forest | one-hot | 0.9647 ± 0.0048 | 0.9175 ± 0.0230 | 0.9180 ± 0.0181 | 0.0898 | 0.1063 |
| Random Forest | ordinal | 0.9821 ± 0.0052 | 0.9576 ± 0.0215 | 0.9633 ± 0.0107 | 0.0681 | 0.1053 |
| SVM | one-hot | 0.9994 ± 0.0013 | 0.9994 ± 0.0015 | 0.9978 ± 0.0049 | 0.0044 | 0.0121 |
| SVM | ordinal | 0.9919 ± 0.0048 | 0.9911 ± 0.0109 | 0.9876 ± 0.0161 | 0.0153 | 0.0179 |

## One-hot minus ordinal

Positive Macro F1 delta favors one-hot; negative Brier delta favors one-hot.

| Model | Δ accuracy | Δ macro F1 | Δ Brier | Δ ECE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | -0.0064 | -0.0178 | +0.0164 | +0.0077 |
| Gradient Boosting | +0.0041 | +0.0079 | -0.0067 | -0.0084 |
| K-Nearest Neighbors | -0.0070 | -0.0226 | +0.0092 | +0.0546 |
| Logistic Regression | +0.2448 | +0.5606 | -0.3057 | -0.0552 |
| Naive Bayes | +0.1308 | +0.4100 | -0.1301 | +0.0223 |
| Random Forest | -0.0174 | -0.0453 | +0.0216 | +0.0010 |
| SVM | +0.0075 | +0.0102 | -0.0109 | -0.0058 |

## Most frequent out-of-fold confusions

- **Decision Tree / one-hot:** acc→unacc (18), unacc→acc (15), good→acc (5)
- **Gradient Boosting / one-hot:** unacc→acc (14), acc→good (5), acc→vgood (2)
- **K-Nearest Neighbors / one-hot:** acc→unacc (75), good→acc (33), vgood→acc (12)
- **Logistic Regression / one-hot:** unacc→acc (48), acc→unacc (36), good→acc (9)
- **Naive Bayes / one-hot:** unacc→acc (242), acc→good (60), acc→vgood (22)
- **Random Forest / one-hot:** unacc→acc (26), acc→unacc (9), good→acc (7)
- **SVM / one-hot:** acc→vgood (1)
- **Decision Tree / ordinal:** unacc→acc (15), acc→unacc (10), good→acc (6)
- **Gradient Boosting / ordinal:** unacc→acc (17), acc→good (5), acc→unacc (2)
- **K-Nearest Neighbors / ordinal:** unacc→acc (43), good→acc (34), acc→unacc (33)
- **Logistic Regression / ordinal:** acc→unacc (288), unacc→acc (97), good→unacc (56)
- **Naive Bayes / ordinal:** acc→unacc (172), acc→vgood (171), unacc→vgood (132)
- **Random Forest / ordinal:** unacc→acc (11), acc→unacc (8), good→acc (6)
- **SVM / ordinal:** unacc→acc (7), acc→unacc (4), acc→vgood (2)
