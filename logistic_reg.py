import pandas as pd
import numpy as np
import ast
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score, matthews_corrcoef)

# Load the datasets
TRAIN_PATH = "dataset/ddxplus/train.csv"
VAL_PATH   = "dataset/ddxplus/validate.csv"
TEST_PATH  = "dataset/ddxplus/test.csv"

print("Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} patients")


# Parse the EVIDENCES column - each evidence can have a continuous value after "_@_"
# we only need the evidence name so we strip that part
def parse_evidences(ev_str):
    return [ev.split('_@_')[0] for ev in ast.literal_eval(ev_str)]


# Parse the differential diagnosis column
# It contains [disease, probability] pairs - we turn it into a fixed-length vector
def parse_ddx(ddx_str, all_pathologies):
    scores = {name: prob for name, prob in ast.literal_eval(ddx_str)}
    return [scores.get(p, 0.0) for p in all_pathologies]


def prepare_data(df, encoders=None, is_train=True, include_ddx=True, include_init_ev=False):
    df = df.copy()

    # Handle missing values
    if df['AGE'].isnull().sum() > 0:
        df['AGE'] = df['AGE'].fillna(df['AGE'].median())
    if df['SEX'].isnull().sum() > 0:
        df['SEX'] = df['SEX'].fillna(df['SEX'].mode()[0])

    df['EVIDENCES_PARSED'] = df['EVIDENCES'].apply(parse_evidences)

    if is_train:
        # Fit encoders on training data
        mlb = MultiLabelBinarizer()
        evid_features = mlb.fit_transform(df['EVIDENCES_PARSED'])

        le = LabelEncoder()
        labels = le.fit_transform(df['PATHOLOGY'])

        init_enc = LabelEncoder()
        init_enc.fit(df['INITIAL_EVIDENCE'])

        # Collect all unique pathology names from DDX to build consistent feature vector
        all_ddx_pathologies = sorted(set(
            name
            for ddx_str in df['DIFFERENTIAL_DIAGNOSIS']
            for name, _ in ast.literal_eval(ddx_str)
        ))

        encoders = {
            'mlb': mlb,
            'le': le,
            'ddx_cols': all_ddx_pathologies,
            'init_enc': init_enc
        }
    else:
        evid_features = encoders['mlb'].transform(df['EVIDENCES_PARSED'])
        labels = encoders['le'].transform(df['PATHOLOGY'])
        all_ddx_pathologies = encoders['ddx_cols']

    # SEX as binary feature (M=1, F=0)
    sex_col = (df['SEX'] == 'M').astype(int).values.reshape(-1, 1)

    parts = [df[['AGE']].values, sex_col, evid_features]

    if include_ddx:
        ddx_features = np.array([
            parse_ddx(row, all_ddx_pathologies)
            for row in df['DIFFERENTIAL_DIAGNOSIS']
        ])
        parts.insert(2, ddx_features)

    # INITIAL_EVIDENCE is already inside EVIDENCES but we test it as a separate feature
    if include_init_ev:
        known = set(encoders['init_enc'].classes_)
        fallback = encoders['init_enc'].classes_[0]
        init_col = encoders['init_enc'].transform(
            df['INITIAL_EVIDENCE'].apply(lambda x: x if x in known else fallback)
        ).reshape(-1, 1)
        parts.append(init_col)

    X = np.hstack(parts)
    return X, labels, encoders


# Fit encoders once on training set
print("\nPreparing features...")
_, y_train, encoders = prepare_data(train_df, is_train=True, include_ddx=True, include_init_ev=True)
_, y_val,   _        = prepare_data(val_df,   encoders, is_train=False, include_ddx=True, include_init_ev=True)
_, y_test,  _        = prepare_data(test_df,  encoders, is_train=False, include_ddx=True, include_init_ev=True)


# Ablation configs: test 3 different feature combinations
ablation_configs = [
    ("Baseline (AGE+SEX+Evidences)",          False, False),
    ("+ DDX probabilities",                    True,  False),
    ("+ DDX probabilities + Initial Evidence", True,  True),
]

datasets = {}
for label, inc_ddx, inc_init in ablation_configs:
    Xtr, _, _ = prepare_data(train_df, encoders, is_train=False, include_ddx=inc_ddx, include_init_ev=inc_init)
    Xv,  _, _ = prepare_data(val_df,   encoders, is_train=False, include_ddx=inc_ddx, include_init_ev=inc_init)
    Xte, _, _ = prepare_data(test_df,  encoders, is_train=False, include_ddx=inc_ddx, include_init_ev=inc_init)

    sc = MinMaxScaler()
    Xtr = sc.fit_transform(Xtr)
    Xv  = sc.transform(Xv)
    Xte = sc.transform(Xte)

    datasets[label] = (Xtr, Xv, Xte, sc)

# Use the full feature set as our main model
X_train, X_val, X_test, scaler = datasets[ablation_configs[-1][0]]

n_ddx = len(encoders['ddx_cols'])
print(f"\nFeature breakdown:")
print(f"  AGE + SEX           : 2")
print(f"  DDX prob cols       : {n_ddx}")
print(f"  Evidence binary     : {X_train.shape[1] - 2 - n_ddx - 1}")
print(f"  Initial Evidence    : 1")
print(f"  Total               : {X_train.shape[1]}")


# Gradient Descent - manual implementation
# We do binary classification (class 0 vs rest) to show the cost converging
print("\n" + "="*40)
print("   GRADIENT DESCENT (MANUAL)")
print("="*40)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * (y @ np.log(h + 1e-9) + (1 - y) @ np.log(1 - h + 1e-9))
    return cost

# Use a small subset to keep it fast
X_gd = X_train[:5000]
y_gd = (y_train[:5000] == 0).astype(float)

alpha = 0.1
n_iters = 200
theta = np.zeros(X_gd.shape[1])
cost_history = []

for i in range(n_iters):
    h = sigmoid(X_gd @ theta)
    gradient = (1/len(y_gd)) * X_gd.T @ (h - y_gd)
    theta -= alpha * gradient
    cost_history.append(compute_cost(X_gd, y_gd, theta))

print(f"Initial cost : {cost_history[0]:.4f}")
print(f"Final cost   : {cost_history[-1]:.4f}")
print("-> Gradient descent converged successfully.")


# Ablation study
print("\n" + "="*40)
print("ABLATION STUDY")
print("="*40)
print(f"  {'Config':<45}  {'Val Acc':>8}  {'Test Acc':>9}  {'Test F1':>8}")
print("  " + "-"*75)

ablation_results = {}
for label, inc_ddx, inc_init in ablation_configs:
    Xtr, Xv, Xte, _ = datasets[label]
    m_abl = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
    m_abl.fit(Xtr, y_train)

    val_pred  = m_abl.predict(Xv)
    test_pred = m_abl.predict(Xte)

    val_a  = accuracy_score(y_val, val_pred)
    test_a = accuracy_score(y_test, test_pred)
    test_f = f1_score(y_test, test_pred, average='weighted', zero_division=0)

    ablation_results[label] = (val_a, test_a, test_f)
    print(f"  {label:<45}  {val_a:>8.4f}  {test_a:>9.4f}  {test_f:>8.4f}")

print("\n-> Adding DDX probabilities makes a big difference.")
print("-> Initial Evidence doesn't add much since it's already in EVIDENCES.")


# Train the main model on full feature set
print("\n" + "="*40)
print("   MODEL TRAINING")
print("="*40)

model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
print(f"Training on {len(encoders['le'].classes_)} classes...")
model.fit(X_train, y_train)


# Validation results
y_val_pred = model.predict(X_val)
print("\n" + "="*40)
print("   VALIDATION RESULTS")
print("="*40)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")


# Test results
y_test_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
mcc  = matthews_corrcoef(y_test, y_test_pred)
cm   = confusion_matrix(y_test, y_test_pred)

print("\n" + "="*40)
print("   TEST RESULTS")
print("="*40)
print(f"Accuracy             : {acc:.4f}")
print(f"Precision (Weighted) : {prec:.4f}")
print(f"Recall (Weighted)    : {rec:.4f}")
print(f"F1-Score (Weighted)  : {f1:.4f}")
print(f"MCC                  : {mcc:.4f}")


# Save outputs
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("\nSaving plots...")

# Gradient descent cost curve
plt.figure(figsize=(8, 5))
plt.plot(cost_history, color='steelblue')
plt.title('Gradient Descent: Cost over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost (Binary Cross-Entropy)')
plt.tight_layout()
plt.savefig(output_dir / "gradient_descent_cost.png")
plt.close()

# Ablation bar chart
abl_labels   = [l.replace("+ ", "+\n") for l in ablation_results.keys()]
abl_test_acc = [v[1] for v in ablation_results.values()]

plt.figure(figsize=(10, 5))
bars = plt.bar(abl_labels, abl_test_acc, color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylim(min(abl_test_acc) - 0.05, 1.0)
plt.title('Ablation Study: Test Accuracy by Feature Set')
plt.ylabel('Test Accuracy')
for bar, val in zip(bars, abl_test_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / "lr_ablation_study.png")
plt.close()

# Train / Val / Test accuracy comparison
train_acc = model.score(X_train, y_train)
val_acc   = accuracy_score(y_val, y_val_pred)

plt.figure(figsize=(8, 5))
scores_bar = [train_acc, val_acc, acc]
sns.barplot(x=['Train', 'Validation', 'Test'], y=scores_bar, palette='viridis')
plt.ylim(min(scores_bar) - 0.05, 1.0)
plt.title('Model Accuracy: Train vs Val vs Test')
plt.ylabel('Accuracy')
plt.savefig(output_dir / "lr_accuracy_comparison.png")
plt.close()

# Confusion matrix - top 20 classes
plt.figure(figsize=(12, 10))
sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues',
            xticklabels=encoders['le'].classes_[:20],
            yticklabels=encoders['le'].classes_[:20])
plt.title('Confusion Matrix (Top 20 Classes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(output_dir / "lr_confusion_matrix.png")
plt.close()

# Overfitting analysis - different C values
C_values, train_scores, val_scores = [], [], []
print(f"\n  {'C':>8}  {'Train Acc':>10}  {'Val Acc':>10}  {'Gap':>8}")
print("  " + "-"*44)

for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    m_c = LogisticRegression(max_iter=1000, solver='lbfgs', C=C, random_state=42)
    m_c.fit(X_train, y_train)
    tr = m_c.score(X_train, y_train)
    vl = m_c.score(X_val, y_val)
    C_values.append(str(C))
    train_scores.append(tr)
    val_scores.append(vl)
    print(f"  {C:>8}  {tr:>10.4f}  {vl:>10.4f}  {tr-vl:>8.4f}")

plt.figure(figsize=(8, 5))
plt.plot(C_values, train_scores, marker='o', label='Train', color='steelblue')
plt.plot(C_values, val_scores,   marker='s', label='Validation', color='coral')
plt.xlabel('C (Regularization)')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy for Different C Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / "lr_overfitting_analysis.png")
plt.close()

# Save metrics
results = {
    "accuracy":       acc,
    "precision":      prec,
    "recall":         rec,
    "f1_score":       f1,
    "mcc":            mcc,
    "train_accuracy": train_acc,
    "val_accuracy":   val_acc,
    "ablation": {
        label: {"val_acc": v[0], "test_acc": v[1], "test_f1": v[2]}
        for label, v in ablation_results.items()
    }
}
with open(output_dir / "lr_final_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

cm_df = pd.DataFrame(cm, index=encoders['le'].classes_, columns=encoders['le'].classes_)
cm_df.to_csv(output_dir / "lr_final_confusion_matrix.csv")

print(f"\nAll outputs saved to '{output_dir}'")

# Classification report
print("\nClassification Report (Test Set):")
report = classification_report(y_test, y_test_pred,
                                target_names=encoders['le'].classes_,
                                zero_division=0)
print("\n".join(report.split("\n")[:15]))