# 📱 Digital Addiction Risk Predictor

## Project Overview

The Digital Addiction Risk Predictor is a Machine Learning project that uses Logistic Regression to analyze smartphone usage patterns and estimate the risk of excessive smartphone usage.

The model classifies usage patterns into two categories:
- Low Risk
- High Risk

This project is designed for educational purposes and demonstrates how machine learning can be used for binary classification and probability-based prediction.

## 🔗 Project Links

- **GitHub Repository:** https://github.com/mhsaee01234/CIE2-Activity2-AD3318-RCA-
- **Kaggle Dataset:** https://www.kaggle.com/datasets/saeesm/smartphone-usage-and-digital-addiction-dataset

##🔗 Submission Link
Clickable Submission Link: https://github.com/complexiota


## Features

- Daily screen time analysis
- Night-time smartphone usage analysis
- App sessions per day
- Social media usage analysis
- Binary classification (Low Risk / High Risk)
- Probability-based predictions
- Adjustable decision threshold
- Confusion matrix visualization
- ROC curve and ROC-AUC evaluation
- L1 and L2 regularization comparison
- Interactive Streamlit web application

## Machine Learning Algorithm

**Logistic Regression**

The model uses four input features:
1. Daily screen time
2. Night-time usage
3. App sessions
4. Social media usage

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- Log Loss
- Confusion Matrix

The recommended decision threshold is 0.3, prioritizing the identification of more potentially high-risk usage patterns.

## Dataset

The project uses a generated smartphone usage dataset containing 4,670 records.

Dataset features include:
- Daily screen time
- Night-time usage
- App sessions
- Social media usage
- Addiction risk label

The dataset is hosted on Kaggle.

## Project Structure

```text
Digital-Addiction-Predictor/
│
├── data/
│   ├── smartphone_usage.csv
│   └── processed/
│
├── models/
│   ├── logistic_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── sigmoid_curve.png
│   ├── decision_behaviour.png
│   ├── model_metrics.csv
│   ├── threshold_comparison.csv
│   └── regularization_comparison.csv
│
├── app.py
├── prepare_data.py
├── train_model.py
├── inspect_dataset.py
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/mhsaee01234/CIE2-Activity2-AD3318-RCA-.git
cd CIE2-Activity2-AD3318-RCA-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

```bash
python prepare_data.py
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Launch the application

```bash
streamlit run app.py
```

## Disclaimer

This is an educational machine-learning project. Its predictions are based on patterns in the generated dataset and do not provide a clinical diagnosis of digital addiction.
