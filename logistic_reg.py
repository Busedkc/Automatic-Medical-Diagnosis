import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, matthews_corrcoef
)

print("  AUTOMATIC MEDICAL DIAGNOSIS - LOGISTIC REGRESSION")
train_df    = pd.read_csv('train.csv')
validate_df = pd.read_csv('validate.csv')
test_df     = pd.read_csv('test.csv')

print("\n[1] Dataset loaded successfully.")
print(f"    Train size    : {len(train_df)} patients")
print(f"    Validate size : {len(validate_df)} patients")
print(f"    Test size     : {len(test_df)} patients")

print("\n--- First 5 rows of training set ---")
print(train_df.head())


# 2. DATASET EXPLORATION

print("  DATASET EXPLORATION")
print("\nData Types")
print(train_df.dtypes)
print("\nBasic Statistics (AGE)")
print(train_df[['AGE']].describe())

print("\n--- Number of unique pathologies ---")
print(f"    {train_df['PATHOLOGY'].nunique()} unique diseases")

print("\n--- Pathology distribution (top 10) ---")
print(train_df['PATHOLOGY'].value_counts().head(10))

print("\n--- SEX distribution ---")
print(train_df['SEX'].value_counts())


# 3. DATA VISUALIZATION


print("\nDATA VISUALIZATION")


# --- 3a. Age distribution ---
plt.figure(figsize=(8, 5))
plt.hist(train_df['AGE'], bins=30, color='steelblue', edgecolor='white')
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=150)
plt.show()
print("    [Saved] age_distribution.png")

# --- 3b. Top 15 pathology counts ---
top_pathologies = train_df['PATHOLOGY'].value_counts().head(15)
plt.figure(figsize=(12, 5))
plt.bar(top_pathologies.index, top_pathologies.values, color='coral', edgecolor='white')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.title('Top 15 Most Common Pathologies (Training Set)')
plt.xlabel('Pathology')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.savefig('pathology_distribution.png', dpi=150)
plt.show()
print("    [Saved] pathology_distribution.png")

# --- 3c. Scatter: Age vs SEX ---
plt.figure(figsize=(8, 5))
sex_numeric = train_df['SEX'].map({'M': 1, 'F': 0})
plt.scatter(train_df['AGE'], sex_numeric, alpha=0.3, c=sex_numeric,
            cmap='coolwarm', marker='o', s=5)
plt.yticks([0, 1], ['Female', 'Male'])
plt.xlabel('Age')
plt.title('Age vs. Sex Distribution')
plt.tight_layout()
plt.savefig('age_sex_scatter.png', dpi=150)
plt.show()
print("    [Saved] age_sex_scatter.png")


# 4. DATA PREPROCESSING & FEATURE ENGINEERING

print("\nDATA PREPROCESSING & FEATURE ENGINEERING")



def parse_evidences(ev_str):
    # Convert string to list, then keep only base evidence code (drop _@_value part)
    ev_list = ast.literal_eval(ev_str)
    return [ev.split('_@_')[0] for ev in ev_list]


def preprocess(df):
    """
    Preprocess a DDXPlus patient DataFrame:
      - Fill missing values
      - Encode SEX (One-Hot encoding, as in One_hot_encoding.py)
      - Keep AGE as numeric
      - Parse EVIDENCES list and binary-encode with MultiLabelBinarizer
    """
    df = df.copy()

    # --- Check for missing values ---
    print(f"\n  Missing values before cleaning:\n{df.isnull().sum()}")

    # Fill missing AGE with median (same approach as multiple_features.py)
    if df['AGE'].isnull().sum() > 0:
        median_age = df['AGE'].median()
        df['AGE'] = df['AGE'].fillna(median_age)
        print(f"  Filled missing AGE values with median: {median_age}")

    # Fill missing SEX with mode
    if df['SEX'].isnull().sum() > 0:
        df['SEX'] = df['SEX'].fillna(df['SEX'].mode()[0])
        print("  Filled missing SEX values with mode.")

    print(f"\n  Missing values after cleaning:\n{df.isnull().sum()}")

    # --- Encode SEX as binary (One-Hot encoding, as in One_hot_encoding.py) ---
    df_encoded = pd.get_dummies(df[['SEX']], columns=['SEX'], dtype=int)
    sex_col = 'SEX_M' if 'SEX_M' in df_encoded.columns else df_encoded.columns[0]
    df['SEX_M'] = df_encoded[sex_col]

    # --- Parse and binary-encode EVIDENCES using MultiLabelBinarizer ---
    mlb = MultiLabelBinarizer()
    ev_matrix = mlb.fit_transform(df['EVIDENCES'].apply(parse_evidences))
    ev_df = pd.DataFrame(ev_matrix, columns=mlb.classes_)

    # Combine all features
    feature_df = pd.concat([df[['AGE', 'SEX_M']].reset_index(drop=True),
                             ev_df.reset_index(drop=True)], axis=1)

    target = df['PATHOLOGY'].reset_index(drop=True)

    return feature_df, target, list(mlb.classes_)


print("\nPreprocessing TRAIN set")
X_train_full, y_train, train_ev = preprocess(train_df)

print("\nPreprocessing VALIDATE set")
X_val_full, y_val, val_ev = preprocess(validate_df)

print("\nPreprocessing TEST set")
X_test_full, y_test, test_ev = preprocess(test_df)

# Align columns: use only evidence codes present in ALL splits
common_ev   = sorted(set(train_ev) & set(val_ev) & set(test_ev))
shared_cols = ['AGE', 'SEX_M'] + common_ev

X_train = X_train_full.reindex(columns=shared_cols, fill_value=0)
X_val   = X_val_full.reindex(columns=shared_cols, fill_value=0)
X_test  = X_test_full.reindex(columns=shared_cols, fill_value=0)

print(f"\n  Total features after preprocessing : {X_train.shape[1]}")
print(f"  (AGE, SEX_M + {len(common_ev)} binary evidence features)")


# 5. FEATURE SCALING (MinMax - as in multiple_features.py)

print("\nFEATURE SCALING (MinMaxScaler)")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("  MinMaxScaler applied. All features now in [0, 1].")


# 6. LOGISTIC REGRESSION MODEL

print("\nLOGISTIC REGRESSION MODEL")


model = LogisticRegression(
    max_iter=1000,   # enough iterations to converge
    solver='lbfgs',  # efficient for multi-class
    C=1.0,           # regularization strength (default)
    random_state=42
)

print("\n  Training the model...")
model.fit(X_train_scaled, y_train)
print("  Training complete.")


# 7. PREDICTIONS & EVALUATION
print("\nEVALUATION")

# --- Validation set ---
y_val_pred = model.predict(X_val_scaled)
val_acc    = accuracy_score(y_val, y_val_pred)
val_mcc    = matthews_corrcoef(y_val, y_val_pred)

print(f"\n  [Validation Set]")
print(f"    Accuracy : {val_acc:.4f}")
print(f"    MCC      : {val_mcc:.4f}")

# --- Test set ---
y_test_pred = model.predict(X_test_scaled)
test_acc    = accuracy_score(y_test, y_test_pred)
test_mcc    = matthews_corrcoef(y_test, y_test_pred)

print(f"\n  [Test Set]")
print(f"    Accuracy : {test_acc:.4f}")
print(f"    MCC      : {test_mcc:.4f}")

# --- Training set score ---
train_acc = model.score(X_train_scaled, y_train)
print(f"\n  [Training Set]")
print(f"    Accuracy : {train_acc:.4f}")

# --- Detailed classification report (precision, recall, F1) ---
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_test_pred, zero_division=0))


# 8. CONFUSION MATRIX

print("\nCONFUSION MATRIX")


top_n       = 15
top_classes = train_df['PATHOLOGY'].value_counts().head(top_n).index.tolist()

mask_test = y_test.isin(top_classes)
mask_pred = pd.Series(y_test_pred).isin(top_classes)
combined  = mask_test & mask_pred.values

y_test_top = y_test[combined]
y_pred_top = pd.Series(y_test_pred)[combined]

cm = confusion_matrix(y_test_top, y_pred_top, labels=top_classes)

fig, ax = plt.subplots(figsize=(14, 11))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_classes)
disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
ax.set_title(f'Confusion Matrix - Top {top_n} Pathologies (Test Set)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("    [Saved] confusion_matrix.png")


# 9. EXAMPLE PREDICTION


print("\nEXAMPLE PREDICTION")


sample_idx   = 0
sample_input = X_test_scaled[sample_idx].reshape(1, -1)
true_label   = y_test.iloc[sample_idx]

predicted_class = model.predict(sample_input)[0]
predicted_proba = model.predict_proba(sample_input)
class_idx       = list(model.classes_).index(predicted_class)
confidence      = predicted_proba[0][class_idx]

print(f"\n  Patient #{sample_idx}")
print(f"    True Pathology      : {true_label}")
print(f"    Predicted Pathology : {predicted_class}")
print(f"    Confidence          : {confidence:.4f} ({confidence*100:.1f}%)")

top3_idx    = np.argsort(predicted_proba[0])[::-1][:3]
top3_labels = [(model.classes_[i], predicted_proba[0][i]) for i in top3_idx]
print("\n  Top 3 predictions:")
for rank, (disease, prob) in enumerate(top3_labels, 1):
    print(f"    {rank}. {disease:40s}  p={prob:.4f}")


# 10. OVERFITTING ANALYSIS

print("\nOVERFITTING ANALYSIS")


C_values     = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores = []
val_scores   = []

print("\n  Testing different regularization strengths (C)...")
print(f"  {'C':>8}  {'Train Acc':>10}  {'Val Acc':>10}  {'Gap':>8}")
print("  " + "-" * 44)

for C in C_values:
    m = LogisticRegression(max_iter=1000, solver='lbfgs', C=C, random_state=42)
    m.fit(X_train_scaled, y_train)
    tr_acc   = m.score(X_train_scaled, y_train)
    val_acc_ = m.score(X_val_scaled, y_val)
    gap      = tr_acc - val_acc_
    train_scores.append(tr_acc)
    val_scores.append(val_acc_)
    print(f"  {C:>8}  {tr_acc:>10.4f}  {val_acc_:>10.4f}  {gap:>8.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(len(C_values)), train_scores, marker='o', label='Train Accuracy', color='steelblue')
plt.plot(range(len(C_values)), val_scores,   marker='s', label='Val Accuracy',   color='coral')
plt.xticks(range(len(C_values)), [str(c) for c in C_values])
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy\n(Overfitting Analysis)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=150)
plt.show()
print("\n    [Saved] overfitting_analysis.png")

max_gap = max(abs(tr - vl) for tr, vl in zip(train_scores, val_scores))
print(f"\n  Maximum train-val gap : {max_gap:.4f}")
print("\n  Interpretation:")
if max_gap < 0.02:
    print("  -> Train and val accuracy are very close across all C values.")
    print("  -> No significant overfitting detected.")
    print("  -> The model generalizes well to unseen data.")
else:
    print("  -> A gap exists at high C values (less regularization).")
    print("  -> Consider using a smaller C to reduce overfitting.")


# 11. SUMMARY

print("\nSUMMARY")
print("=" * 60)
print(f"  Model            : Logistic Regression (lbfgs, C=1.0)")
print(f"  Features         : {X_train.shape[1]}")
print(f"  Classes          : {len(model.classes_)} pathologies")
print(f"  Train Accuracy   : {train_acc:.4f}")
print(f"  Val Accuracy     : {val_acc:.4f}")
print(f"  Test Accuracy    : {test_acc:.4f}")
print(f"  Test MCC         : {test_mcc:.4f}")
print("=" * 60)


#print(X_train.columns.tolist())

