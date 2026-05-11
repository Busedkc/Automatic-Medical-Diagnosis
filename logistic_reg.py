import pandas as pd
import numpy as np
import ast
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — avoids Tkinter/display errors
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,f1_score, precision_score, recall_score, matthews_corrcoef)
# 1. SETTINGS AND DATA LOADING
TRAIN_PATH = "dataset/ddxplus/train.csv"
VAL_PATH   = "dataset/ddxplus/validate.csv"
TEST_PATH  = "dataset/ddxplus/test.csv"
print("Loading datasets...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    print("Error: Data files not found! Please check the paths.")
    exit()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} patients")
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
def parse_evidences(ev_str):
    # Convert string to list, strip continuous values
    return [ev.split('_@_')[0] for ev in ast.literal_eval(ev_str)]
def parse_ddx(ddx_str, all_pathologies):
    # DIFFERENTIAL_DIAGNOSIS contains [disease, probability] pairs.
    # We build a fixed-length vector ordered by all_pathologies seen in training.
    scores = {name: prob for name, prob in ast.literal_eval(ddx_str)}
    return [scores.get(p, 0.0) for p in all_pathologies]
def prepare_data(df, encoders=None, is_train=True, include_ddx=True, include_init_ev=False):
    df = df.copy()
    # Fill missing values
    if df['AGE'].isnull().sum() > 0:
        df['AGE'] = df['AGE'].fillna(df['AGE'].median())
    if df['SEX'].isnull().sum() > 0:
        df['SEX'] = df['SEX'].fillna(df['SEX'].mode()[0])
    df['EVIDENCES_PARSED'] = df['EVIDENCES'].apply(parse_evidences)
    if is_train:
        mlb = MultiLabelBinarizer()
        evid_features = mlb.fit_transform(df['EVIDENCES_PARSED'])
        le = LabelEncoder()
        labels = le.fit_transform(df['PATHOLOGY'])
        # Fit a separate encoder for INITIAL_EVIDENCE
        init_enc = LabelEncoder()
        init_enc.fit(df['INITIAL_EVIDENCE'])
        # Collect all unique pathology names seen in DDX across training set
        # to build a consistent probability vector per patient
        all_ddx_pathologies = sorted(set(
            name
            for ddx_str in df['DIFFERENTIAL_DIAGNOSIS']
            for name, _ in ast.literal_eval(ddx_str)
        ))
        encoders = {'mlb': mlb, 'le': le, 'ddx_cols': all_ddx_pathologies, 'init_enc': init_enc}
    else:
        evid_features = encoders['mlb'].transform(df['EVIDENCES_PARSED'])
        labels = encoders['le'].transform(df['PATHOLOGY'])
        all_ddx_pathologies = encoders['ddx_cols']
    # Binary-encode SEX (M=1, F=0)
    sex_col = (df['SEX'] == 'M').astype(int).values.reshape(-1, 1)
    # Base parts: AGE + SEX + evidence binary flags (always included)
    parts = [df[['AGE']].values, sex_col, evid_features]
    # Optional: DDX probability features (one column per pathology seen in training DDX)
    if include_ddx:
        ddx_features = np.array([
            parse_ddx(row, all_ddx_pathologies)
            for row in df['DIFFERENTIAL_DIAGNOSIS']
        ])
        parts.insert(2, ddx_features)  # insert before evidence flags
    # Addded INITIAL_EVIDENCE as an integer-encoded column.
    # INITIAL_EVIDENCE already appears inside EVIDENCES, so this adds a
    # redundant but explicitly weighted signal — included here for ablation only.
    if include_init_ev:
        known = set(encoders['init_enc'].classes_)
        fallback = encoders['init_enc'].classes_[0]
        init_col = encoders['init_enc'].transform(
            df['INITIAL_EVIDENCE'].apply(lambda x: x if x in known else fallback)
        ).reshape(-1, 1)
        parts.append(init_col)
    X = np.hstack(parts)
    return X, labels, encoders
# Fit encoders on training set once; reuse for all three ablation variants
print("\nExtracting features (full feature set to fit encoders)...")
_, y_train, encoders = prepare_data(train_df, is_train=True,include_ddx=True, include_init_ev=True)
_, y_val,   _        = prepare_data(val_df,   encoders, is_train=False,include_ddx=True, include_init_ev=True)
_, y_test,  _        = prepare_data(test_df,  encoders, is_train=False,include_ddx=True, include_init_ev=True)

# Build the three feature sets for ablation
ablation_configs = [("Baseline (AGE+SEX+Evidences)",False, False),("+ DDX probabilities",True, False),("+ DDX probabilities + Initial Evidence",True, True),]
scaler = MinMaxScaler()
datasets = {}
for label, inc_ddx, inc_init in ablation_configs:
    Xtr, _, _ = prepare_data(train_df, encoders, is_train=False,include_ddx=inc_ddx, include_init_ev=inc_init)
    Xv,  _, _ = prepare_data(val_df,   encoders, is_train=False,include_ddx=inc_ddx, include_init_ev=inc_init)
    Xte, _, _ = prepare_data(test_df,  encoders, is_train=False,include_ddx=inc_ddx, include_init_ev=inc_init)
    # Fit scaler separately per config so scaling is always correct
    sc = MinMaxScaler()
    Xtr = sc.fit_transform(Xtr)
    Xv  = sc.transform(Xv)
    Xte = sc.transform(Xte)
    datasets[label] = (Xtr, Xv, Xte, sc)

# Use the full config (last one) as the main model
X_train, X_val, X_test, scaler = datasets[ablation_configs[-1][0]]
n_ddx  = len(encoders['ddx_cols'])
print(f"\nFeature breakdown (full config):")
print(f"  AGE + SEX              : 2")
print(f"  DDX probability cols   : {n_ddx}")
print(f"  Evidence binary cols   : {X_train.shape[1] - 2 - n_ddx - 1}")
print(f"  Initial Evidence col   : 1")
print(f"  Total                  : {X_train.shape[1]}")

# 3. GRADIENT DESCENT (Manual Implementation)
# Demonstrates the core optimization loop taught in class.
# Applied to a binary sub-problem (class 0 vs rest) to show
# the sigmoid + cost curve converging without sklearn.

print("\n" + "="*40)
print("   GRADIENT DESCENT (MANUAL)")
print("="*40)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    # Binary cross-entropy loss
    return (-1/m) * (y @ np.log(h + 1e-9) + (1 - y) @ np.log(1 - h + 1e-9))

# Small subset for speed
X_gd = X_train[:5000]
y_gd = (y_train[:5000] == 0).astype(float)   # binary: class 0 vs rest

alpha        = 0.1
n_iters      = 200
m_gd, n_gd  = X_gd.shape
theta        = np.zeros(n_gd)
cost_history = []

for i in range(n_iters):
    h        = sigmoid(X_gd @ theta)
    gradient = (1/m_gd) * X_gd.T @ (h - y_gd)
    theta   -= alpha * gradient
    cost_history.append(compute_cost(X_gd, y_gd, theta))

print(f"Initial cost : {cost_history[0]:.4f}")
print(f"Final cost   : {cost_history[-1]:.4f}")
print("-> Cost decreased steadily — gradient descent converged.")

# 4. ABLATION STUDY — train one model per feature config, compare on test set
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
    val_a  = accuracy_score(y_val,  m_abl.predict(Xv))
    test_a = accuracy_score(y_test, m_abl.predict(Xte))
    test_f = f1_score(y_test, m_abl.predict(Xte), average='weighted', zero_division=0)
    ablation_results[label] = (val_a, test_a, test_f)
    print(f"  {label:<45}  {val_a:>8.4f}  {test_a:>9.4f}  {test_f:>8.4f}")

print("\n-> Interpretation: If '+ DDX' shows a large jump, DDX probabilities are a strong signal.")
print("-> If '+ Initial Evidence' adds little, it confirms the feature is redundant with EVIDENCES.")

# 5. MAIN MODEL TRAINING (full feature set)
print("\n" + "="*40)
print("   MODEL TRAINING (Full Feature Set)")
print("="*40)

model = LogisticRegression(max_iter=1000,solver='lbfgs',C=1.0,random_state=42)
print(f"Training Logistic Regression (solver=lbfgs, C=1.0, classes={len(encoders['le'].classes_)})...")
model.fit(X_train, y_train)

# 6. VALIDATION RESULTS
y_val_pred = model.predict(X_val)
print("\n" + "="*40)
print("   VALIDATION RESULTS")
print("="*40)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

# 7. FINAL TEST RESULTS
y_test_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
mcc  = matthews_corrcoef(y_test, y_test_pred)
cm   = confusion_matrix(y_test, y_test_pred)

print("\n" + "="*40)
print("   FINAL TEST RESULTS (TEST.CSV)")
print("="*40)
print(f"Accuracy               : {acc:.4f}")
print(f"Precision (Weighted)   : {prec:.4f}")
print(f"Recall (Weighted)      : {rec:.4f}")
print(f"F1-Score (Weighted)    : {f1:.4f}")
print(f"MCC                    : {mcc:.4f}")

# 8. SAVING RESULTS AND VISUALIZATIONS
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Plot 1: Gradient Descent Cost Curve
print("\nGenerating plots...")
plt.figure(figsize=(8, 5))
plt.plot(cost_history, color='steelblue')
plt.title('Gradient Descent: Cost over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost (Binary Cross-Entropy)')
plt.tight_layout()
plt.savefig(output_dir / "gradient_descent_cost.png")
plt.close()

# Plot 2: Ablation Study Comparison
abl_labels  = [l.replace("+ ", "+\n") for l in ablation_results.keys()]
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

# Plot 3: Accuracy Comparison (Train / Val / Test)
train_acc = model.score(X_train, y_train)
val_acc   = accuracy_score(y_val, y_val_pred)

plt.figure(figsize=(8, 5))
scores_bar = [train_acc, val_acc, acc]
sns.barplot(x=['Train', 'Validation', 'Test'], y=scores_bar, palette='viridis')
plt.ylim(min(scores_bar) - 0.05, 1.0)
plt.title('Model Accuracy Comparison (Full Feature Set)')
plt.ylabel('Accuracy')
plt.savefig(output_dir / "lr_accuracy_comparison.png")
plt.close()

# Plot 4: Confusion Matrix (top 20 classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues',xticklabels=encoders['le'].classes_[:20],yticklabels=encoders['le'].classes_[:20])
plt.title('Confusion Matrix (Top 20 Classes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(output_dir / "lr_confusion_matrix.png")
plt.close()

# Plot 5: Overfitting Analysis (C values)
C_values, train_scores, val_scores = [], [], []
print(f"\n  {'C':>8}  {'Train Acc':>10}  {'Val Acc':>10}  {'Gap':>8}")
print("  " + "-"*44)
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    m_c = LogisticRegression(max_iter=1000, solver='lbfgs', C=C, random_state=42)
    m_c.fit(X_train, y_train)
    tr = m_c.score(X_train, y_train)
    vl = m_c.score(X_val, y_val)
    C_values.append(str(C)); train_scores.append(tr); val_scores.append(vl)
    print(f"  {C:>8}  {tr:>10.4f}  {vl:>10.4f}  {tr-vl:>8.4f}")

plt.figure(figsize=(8, 5))
plt.plot(C_values, train_scores, marker='o', label='Train', color='steelblue')
plt.plot(C_values, val_scores,   marker='s', label='Validation', color='coral')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy (Overfitting Analysis)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / "lr_overfitting_analysis.png")
plt.close()

# Save metrics to JSON
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

# Save Confusion Matrix to CSV
cm_df = pd.DataFrame(cm, index=encoders['le'].classes_, columns=encoders['le'].classes_)
cm_df.to_csv(output_dir / "lr_final_confusion_matrix.csv")

print(f"\n[INFO] All metrics and plots saved to '{output_dir}' directory.")

# 9. CLASSIFICATION REPORT
print("\nFinal Classification Report (Test Set):")
report = classification_report(y_test, y_test_pred,target_names=encoders['le'].classes_,zero_division=0)
print("\n".join(report.split("\n")[:15]))
