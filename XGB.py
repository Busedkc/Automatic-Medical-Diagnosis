import pandas as pd
import matplotlib.pyplot as plt
import ast
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
from scipy.sparse import hstack

print("----- XGBOOST MODEL -----\n")

# Load Datasets
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("validate.csv")
test_df  = pd.read_csv("test.csv")

print(train_df.head(),"\n")
print(train_df.isnull().sum(),"\n")

# We exclude DIFFERENTIAL_DIAGNOSIS from the input features
# This column already contains possible disease names and probabilities'
# so using it could cause data leakage and make the model unfairly strong
# Instead, the model only uses demographic information and symptom evidence
feature_cols = ["AGE", "SEX", "INITIAL_EVIDENCE", "EVIDENCES"]

# Separate the input features from the target label
# X contains the information used to make predictions
X_train_df = train_df[feature_cols].copy()
X_val_df   = val_df[feature_cols].copy()
X_test_df  = test_df[feature_cols].copy()
# y contains the correct disease name that the model should learn to predict.
y_train_text = train_df["PATHOLOGY"]
y_val_text   = val_df["PATHOLOGY"]
y_test_text  = test_df["PATHOLOGY"]

# Encode target label
# XGBoost cannot directly learn from disease names written as text.
# LabelEncoder converts each disease name into a numeric class.
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_val   = label_encoder.transform(y_val_text)
y_test  = label_encoder.transform(y_test_text)

# Convert EVIDENCES into List
# ast.literal_eval safely converts the string into a real Python list.
def parse_evidences(x):
    return ast.literal_eval(x) if pd.notna(x) else []

X_train_df["EVIDENCES"] = X_train_df["EVIDENCES"].apply(parse_evidences)
X_val_df["EVIDENCES"]   = X_val_df["EVIDENCES"].apply(parse_evidences)
X_test_df["EVIDENCES"]  = X_test_df["EVIDENCES"].apply(parse_evidences)

# Use Multi-Label Binarization for evidences
# Each patient may have multiple evidence codes
# MultiLabelBinarizer converts the lists into binary columns
mlb = MultiLabelBinarizer(sparse_output=True)

X_train_evidences = mlb.fit_transform(X_train_df["EVIDENCES"])
X_val_evidences   = mlb.transform(X_val_df["EVIDENCES"])
X_test_evidences  = mlb.transform(X_test_df["EVIDENCES"])

# Features that require categorical encoding
# These columns contain one value per patient, so OneHotEncoder is suitable.
normal_cols = ["AGE", "SEX", "INITIAL_EVIDENCE"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

# Fit encoder on training data
X_train_normal = encoder.fit_transform(X_train_df[normal_cols])
X_val_normal   = encoder.transform(X_val_df[normal_cols])
X_test_normal  = encoder.transform(X_test_df[normal_cols])

# Combine the encoded normal features and encoded evidence features
# hstack joins them side by side into one final input matrix
# The final matrix is still sparse which keeps memory usage lower
X_train = hstack([X_train_normal, X_train_evidences])
X_val   = hstack([X_val_normal, X_val_evidences])
X_test  = hstack([X_test_normal, X_test_evidences])

# Display final matrix sizes
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape, "\n")

# Build XGB Model
xgb = XGBClassifier(
    # This is a multi-class classification problem
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    max_depth=5,
    learning_rate=0.05,
    # The maximum number of trees
    n_estimators=800,
    # Each tree is trained using 80% of the training rows
    subsample=0.8,
    # Each tree is trained using 80% of the features
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    # Make training faster on a large dataset
    tree_method="hist",
    random_state=42,
    # Stop training if validation loss does not improve for 30 rounds
    early_stopping_rounds=30
)
# Using subsample and colsample_bytree makes the trees slightly different from each other and helps reduce overfitting
# The model learns more general patterns instead of memorizing the full dataset

# Train the model and print the validation loss every 50 boosting rounds
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# Prediction
y_pred       = xgb.predict(X_test)
# Train and validation predictions are used to check generalization.
y_train_pred = xgb.predict(X_train)
y_val_pred   = xgb.predict(X_val)

# Calculate accuracy for train, validation, and test datasets
# Similar values across these three datasets indicate that the model is not strongly overfitting
train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val, y_val_pred)
test_acc  = accuracy_score(y_test, y_pred)

# Calculate evaluation metrics on the test set.
# Weighted averaging is used because the disease classes are not equally represented
# zero_division=0 prevents warnings when the model does not predict a class at all
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
# Show which classes were predicted correctly or confused.
cm = confusion_matrix(y_test, y_pred)
# Give precision, recall, and F1-score for each disease.
cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

# Results
print("\n---- XGBOOST RESULTS ----")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}\n")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(cr)

# Visualize the confusion matrix
plt.figure(figsize=(18, 16))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Number of samples")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot train, validation, and test accuracies
# If the three points are close, the model generalizes well
plt.figure(figsize=(6, 4))
plt.plot(["Train Dataset", "Validation Dataset", "Test Dataset"], [train_acc, val_acc, test_acc], marker="o", color = 'red')
plt.title("Accuracy Comparison for XGB")
plt.ylim(0.98, 1.00)
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Importance plot
# Create readable feature names for the importance plot

# OneHotEncoder gives names for AGE, SEX, and INITIAL_EVIDENCE
normal_feature_names = encoder.get_feature_names_out(normal_cols)
# MultiLabelBinarizer gives names for evidence codes
evidence_feature_names = mlb.classes_
all_feature_names = list(normal_feature_names) + list(evidence_feature_names)

# Store feature importance scores in a dataframe
importance_df = pd.DataFrame({"Feature": all_feature_names, "Importance": xgb.feature_importances_})

# Select top 20 most important features
top20 = importance_df.sort_values(by="Importance", ascending=True).tail(20)

plt.figure(figsize=(12, 8))
plt.barh(top20["Feature"], top20["Importance"])
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 20 Most Important Features")
plt.show()

# Plot one tree from the XGBoost model

# It is only one tree so it does not explain the full model by itself,
# but it helps visualize the tree-based structure of XGBoost.
plot_tree(xgb, tree_idx=0)
plt.title("XGBoost Tree 0")
plt.show()
