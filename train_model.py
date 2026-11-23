
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    log_loss
)

from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. CREATE OUTPUT FOLDERS
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =====================================================
# 2. LOAD PROCESSED DATA
# =====================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()

# =====================================================
# 3. SCALE DATA
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 4. TRAIN LOGISTIC REGRESSION
# =====================================================

model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =====================================================
# 5. GENERATE PREDICTIONS
# =====================================================

y_prob = model.predict_proba(X_test_scaled)[:, 1]

y_pred = (y_prob >= 0.5).astype(int)

# =====================================================
# 6. EVALUATION FUNCTION
# =====================================================

def evaluate_model(y_true, predictions, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Accuracy": accuracy_score(
            y_true, predictions
        ),

        "Precision": precision_score(
            y_true, predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true, predictions,
            zero_division=0
        ),

        "Specificity": specificity,

        "F1-score": f1_score(
            y_true, predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true, probabilities
        ),

        "Log Loss": log_loss(
            y_true, probabilities
        )
    }


# =====================================================
# 7. PRINT MODEL RESULTS
# =====================================================

print("\n========== LOGISTIC REGRESSION RESULTS ==========")

results = evaluate_model(
    y_test,
    y_pred,
    y_prob
)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


# =====================================================
# 8. CONFUSION MATRIX
# =====================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    [0, 1],
    ["Low Risk", "High Risk"]
)

plt.yticks(
    [0, 1],
    ["Low Risk", "High Risk"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=150
)

plt.close()


# =====================================================
# 9. CLASSIFICATION REPORT
# =====================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low Risk", "High Risk"],
        zero_division=0
    )
)


# =====================================================
# 10. MODEL COEFFICIENTS
# =====================================================

print("\n========== MODEL COEFFICIENTS ==========")

for feature, coefficient in zip(
    X_train.columns,
    model.coef_[0]
):
    print(f"{feature}: {coefficient:.4f}")

print("Intercept:", model.intercept_[0])


# =====================================================
# 11. THRESHOLD COMPARISON
# =====================================================

threshold_results = []

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in [0.3, 0.5, 0.7]:

    predictions = (
        y_prob >= threshold
    ).astype(int)

    metrics = evaluate_model(
        y_test,
        predictions,
        y_prob
    )

    threshold_results.append({
        "Threshold": threshold,
        **metrics
    })

    print(f"\nThreshold: {threshold}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    "outputs/threshold_comparison.csv",
    index=False
)


# =====================================================
# 12. DECISION BEHAVIOUR GRAPH
# =====================================================

plt.figure(figsize=(7, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1-score"],
    marker="o",
    label="F1-score"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")

plt.title(
    "Decision Behaviour at Different Thresholds"
)

plt.ylim(0, 1.05)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/decision_behaviour.png",
    dpi=150
)

plt.close()


# =====================================================
# 13. MAJORITY CLASS BASELINE
# =====================================================

print("\n========== MAJORITY CLASS BASELINE ==========")

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(
    X_train_scaled,
    y_train
)

baseline_pred = baseline.predict(
    X_test_scaled
)

print(
    "Baseline accuracy:",
    accuracy_score(y_test, baseline_pred)
)

print(
    "Model accuracy:",
    accuracy_score(y_test, y_pred)
)


# =====================================================
# 14. L1 VS L2 REGULARIZATION
# =====================================================

print("\n========== REGULARIZATION COMPARISON ==========")

regularization_results = []

for penalty in ["l1", "l2"]:

    regularized_model = LogisticRegression(
        penalty=penalty,
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    regularized_model.fit(
        X_train_scaled,
        y_train
    )

    predictions = regularized_model.predict(
        X_test_scaled
    )

    score = accuracy_score(
        y_test,
        predictions
    )

    regularization_results.append({
        "Regularization": penalty.upper(),
        "Accuracy": score,
        "Nonzero coefficients": int(
            np.count_nonzero(
                regularized_model.coef_
            )
        )
    })

    print(
        penalty.upper(),
        "Accuracy:",
        round(score, 4),
        "Nonzero coefficients:",
        np.count_nonzero(
            regularized_model.coef_
        )
    )


pd.DataFrame(
    regularization_results
).to_csv(
    "outputs/regularization_comparison.csv",
    index=False
)


# =====================================================
# 15. ROC CURVE
# =====================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC={results['ROC-AUC']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png",
    dpi=150
)

plt.close()


# =====================================================
# 16. SIGMOID CURVE
# =====================================================

z = np.linspace(-10, 10, 200)

sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(7, 5))

plt.plot(z, sigmoid)

plt.axhline(
    0.5,
    linestyle="--"
)

plt.xlabel("Log-Odds (z)")
plt.ylabel("Probability")

plt.title(
    "Logistic Regression Sigmoid Function"
)

plt.tight_layout()

plt.savefig(
    "outputs/sigmoid_curve.png",
    dpi=150
)

plt.close()


# =====================================================
# 17. SAVE MODEL EVALUATION METRICS
# =====================================================

metrics_df = pd.DataFrame(
    list(results.items()),
    columns=["Metric", "Score"]
)

metrics_df.to_csv(
    "outputs/model_metrics.csv",
    index=False
)


# =====================================================
# 18. SAVE TRAINED MODEL AND SCALER
# =====================================================

joblib.dump(
    model,
    "models/logistic_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)


# =====================================================
# 19. COMPLETION MESSAGE
# =====================================================

print("\n========== FILES GENERATED ==========")

print("Confusion matrix saved!")
print("Decision behaviour saved!")
print("Model metrics saved!")
print("ROC curve saved!")
print("Sigmoid curve saved!")
print("Threshold comparison saved!")
print("Regularization comparison saved!")

print("Model saved successfully!")
print("Scaler saved successfully!")

print("\nMODEL TRAINING COMPLETED!")
    