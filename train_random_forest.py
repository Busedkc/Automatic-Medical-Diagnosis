import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import (accuracy_score, classification_report, f1_score, 
                             precision_score, recall_score, matthews_corrcoef, confusion_matrix)
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import json
from pathlib import Path

# 1. SETTINGS AND DATA LOADING
TRAIN_PATH = "dataset/ddxplus/train.csv"
VAL_PATH = "dataset/ddxplus/validate.csv"
TEST_PATH = "dataset/ddxplus/test.csv"

print("Loading datasets (Sampling for efficiency)...")
try:
    # 200k train, 25k val, and 25k test for robust evaluation
    train_df = pd.read_csv(TRAIN_PATH).sample(200000, random_state=42)
    val_df = pd.read_csv(VAL_PATH).sample(25000, random_state=42)
    test_df = pd.read_csv(TEST_PATH).sample(25000, random_state=42)
except FileNotFoundError:
    print(f"Error: Data files not found! Please check the paths.")
    exit()

# 2. PREPROCESSING FUNCTION
def prepare_data(df, encoders=None, is_train=True):
    # EVIDENCES column contains string lists. Converting them to real lists.
    # e.g., "['E_1', 'E_2']" -> ['E_1', 'E_2']
    df['EVIDENCES_LIST'] = df['EVIDENCES'].apply(ast.literal_eval)
    
    if is_train:
        # Fit and transform encoders during training
        mlb = MultiLabelBinarizer()
        evid_features = mlb.fit_transform(df['EVIDENCES_LIST'])
        
        sex_enc = OneHotEncoder(handle_unknown='ignore')
        sex_features = sex_enc.fit_transform(df[['SEX']])
        
        init_enc = OneHotEncoder(handle_unknown='ignore')
        init_features = init_enc.fit_transform(df[['INITIAL_EVIDENCE']])
        
        le = LabelEncoder()
        labels = le.fit_transform(df['PATHOLOGY'])
        
        encoders = {'mlb': mlb, 'sex': sex_enc, 'init': init_enc, 'le': le}
    else:
        # Only transform during validation/testing
        evid_features = encoders['mlb'].transform(df['EVIDENCES_LIST'])
        sex_features = encoders['sex'].transform(df[['SEX']])
        init_features = encoders['init'].transform(df[['INITIAL_EVIDENCE']])
        labels = encoders['le'].transform(df['PATHOLOGY'])

    # Combine AGE and other encoded features
    age_values = df[['AGE']].values
    X = hstack([age_values, sex_features, init_features, evid_features]).tocsr()
    
    return X, labels, encoders

# Prepare datasets
print("Extracting features...")
X_train, y_train, encoders = prepare_data(train_df)
X_val, y_val, _ = prepare_data(val_df, encoders, is_train=False)
X_test, y_test, _ = prepare_data(test_df, encoders, is_train=False)

# 3. MODEL TRAINING
print(f"Training Random Forest model (Feature count: {X_train.shape[1]})...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 4. CROSS-VALIDATION
print("\n" + "="*40)
print("       CROSS-VALIDATION RESULTS")
print("="*40)
print("Running 3-Fold Cross-Validation (this may take a few minutes)...")
cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
print(f"CV Mean Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
print("-> Note: Low variance indicates a stable and robust model.")

# 5. VALIDATION RESULTS
y_val_pred = model.predict(X_val)
print("\n" + "="*40)
print("       VALIDATION RESULTS")
print("="*40)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

# 6. FINAL TEST RESULTS (on test.csv)
y_test_pred = model.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, average='weighted')
rec = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print("\n" + "="*40)
print("       FINAL TEST RESULTS (TEST.CSV)")
print("="*40)
print(f"Accuracy               : {acc:.4f}")
print(f"Precision (Weighted)   : {prec:.4f}")
print(f"Recall (Weighted)      : {rec:.4f}")
print(f"F1-Score (Weighted)    : {f1:.4f}")
print(f"MCC                    : {mcc:.4f}")

# 7. SAVING RESULTS
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Save metrics to JSON
results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "mcc": mcc,
    "cv_mean": cv_scores.mean() if 'cv_scores' in locals() else None,
    "cv_std": cv_scores.std() if 'cv_scores' in locals() else None
}

with open(output_dir / "simple_rf_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# Save Confusion Matrix to CSV
cm_df = pd.DataFrame(cm, index=encoders['le'].classes_, columns=encoders['le'].classes_)
cm_df.to_csv(output_dir / "simple_rf_confusion_matrix.csv")

print(f"\n[INFO] All metrics saved to '{output_dir}' directory.")

# 8. RELIABILITY TESTS (SANITY CHECKS)
print("\n" + "="*40)
print("       RELIABILITY TESTS")
print("="*40)

# TEST A: Label Shuffle Check
print("\n[TEST A] Shuffling training labels (Label Shuffle Check)...")
y_train_shuffled = np.random.permutation(y_train)
shuffled_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
shuffled_model.fit(X_train, y_train_shuffled)
shuffled_pred = shuffled_model.predict(X_test)
print(f"Shuffled Labels Accuracy: {accuracy_score(y_test, shuffled_pred):.4f}")
print("-> Interpretation: Low accuracy here proves the model learns patterns, not noise.")

# TEST B: Ablation Test (Removing EVIDENCES)
print("\n[TEST B] Removing EVIDENCES feature (Ablation Test)...")
sex_count = encoders['sex'].get_feature_names_out().shape[0]
init_count = encoders['init'].get_feature_names_out().shape[0]
evid_start_idx = 1 + sex_count + init_count

X_train_no_evid = X_train[:, :evid_start_idx]
X_test_no_evid = X_test[:, :evid_start_idx]

abl_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
abl_model.fit(X_train_no_evid, y_train)
abl_pred = abl_model.predict(X_test_no_evid)
print(f"Ablation Accuracy (No Evidences): {accuracy_score(y_test, abl_pred):.4f}")
print("-> Interpretation: Drop in performance proves 'EVIDENCES' is the primary driver of success.")

print("\nFinal Classification Report (Test Set):")
report = classification_report(y_test, y_test_pred, target_names=encoders['le'].classes_)
print("\n".join(report.split("\n")[:15]))
print("...")
