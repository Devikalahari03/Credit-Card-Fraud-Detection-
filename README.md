# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using logistic regression on a highly imbalanced real-world dataset.

---

## 📌 Project Overview

Credit card fraud is a major financial challenge. This project builds a binary classification model to distinguish between legitimate and fraudulent transactions, with a focus on handling severe class imbalance and minimizing false negatives (missed fraud).

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | **87%** |
| AUC-ROC Score | **94%** |
| Model | Logistic Regression |

---

## 🗂️ Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud rate:** Only 0.17% of transactions are fraudulent (highly imbalanced)
- **Features:** 30 features (V1–V28 from PCA, Time, Amount)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF?style=flat-square)

---

## 🔍 Approach

**1. Exploratory Data Analysis (EDA)**
- Analyzed class imbalance (fraud vs. non-fraud distribution)
- Visualized transaction amount patterns and time-based trends
- Correlation heatmap to identify key features

**2. Data Preprocessing**
- Feature scaling using `StandardScaler` on `Amount` and `Time`
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance
- Train-test split: 80/20

**3. Model Building**
- Trained a **Logistic Regression** classifier
- Evaluated using accuracy, precision, recall, F1-score, and AUC-ROC

**4. Evaluation & Visualization**
- Confusion matrix to analyze true/false positives and negatives
- ROC curve to visualize classifier performance
- Focus on minimizing false negatives (undetected fraud)

---

- ## 📈 Key Visualizations

- Class distribution before and after SMOTE
- Confusion Matrix
- ROC Curve (AUC = 0.94)
- Feature correlation heatmap

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook with full analysis
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/Devikalahari03/Credit-Card-Fraud-Detection-.git
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder.

4. Open and run `Credit_Card_Fraud_Detection.ipynb` in Jupyter Notebook or Google Colab.

---

## 💡 Key Learnings

- How to handle severely imbalanced datasets using SMOTE
- Importance of AUC-ROC over accuracy for imbalanced classification problems
- Logistic regression as a strong, interpretable baseline for fraud detection
- Communicating model performance through visualizations for non-technical audiences

---

## 👩‍💻 Author

**Devika Lahari Bandi**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devika-lahari/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/Devikalahari03)
