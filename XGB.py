import pandas as pd
import matplotlib.pyplot as plt
import ast
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
from scipy.sparse import hstack

print("----- XGBOOST MODEL -----\n")

# Load Data
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("validate.csv")
test_df  = pd.read_csv("test.csv")

# DIFFERENTIAL_DIAGNOSIS is excluded because it contains candidate diseases.
feature_cols = ["AGE", "SEX", "INITIAL_EVIDENCE", "EVIDENCES"]

X_train_df = train_df[feature_cols].copy()
X_val_df   = val_df[feature_cols].copy()
X_test_df  = test_df[feature_cols].copy()

y_train_text = train_df["PATHOLOGY"]
y_val_text   = val_df["PATHOLOGY"]
y_test_text  = test_df["PATHOLOGY"]

# Encode target label
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_val   = label_encoder.transform(y_val_text)
y_test  = label_encoder.transform(y_test_text)

# Convert EVIDENCES TEXT into LIST
def parse_evidences(x):
    return ast.literal_eval(x) if pd.notna(x) else []

X_train_df["EVIDENCES"] = X_train_df["EVIDENCES"].apply(parse_evidences)
X_val_df["EVIDENCES"]   = X_val_df["EVIDENCES"].apply(parse_evidences)
X_test_df["EVIDENCES"]  = X_test_df["EVIDENCES"].apply(parse_evidences)

# Use Multi-Label Binarization for evidences
mlb = MultiLabelBinarizer(sparse_output=True)

X_train_evidences = mlb.fit_transform(X_train_df["EVIDENCES"])
X_val_evidences   = mlb.transform(X_val_df["EVIDENCES"])
X_test_evidences  = mlb.transform(X_test_df["EVIDENCES"])

# Use One-Hot Encoding for other features
normal_cols = ["AGE", "SEX", "INITIAL_EVIDENCE"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

X_train_normal = encoder.fit_transform(X_train_df[normal_cols])
X_val_normal   = encoder.transform(X_val_df[normal_cols])
X_test_normal  = encoder.transform(X_test_df[normal_cols])

# Combine Features
X_train = hstack([X_train_normal, X_train_evidences])
X_val   = hstack([X_val_normal, X_val_evidences])
X_test  = hstack([X_test_normal, X_test_evidences])

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Build Model
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    max_depth=5,
    learning_rate=0.05,
    n_estimators=800,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=30
)

# Train Model
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# Prediction
y_pred = xgb.predict(X_test)

# Evaulation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

# Results
print("\n---- XGBOOST RESULTS ----")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(cr)

plt.figure(figsize=(16, 16))

plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()
