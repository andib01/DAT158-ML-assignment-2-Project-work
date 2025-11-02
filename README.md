This is the code on kaggle that we used to train the model:

import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report

data = pd.read_csv("/kaggle/input/mushroom/agaricus-lepiota.data")

data.columns = [
    "class",
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color",
    "ring-number", "ring-type",
    "spore-print-color", "population", "habitat"
]

print("Data shape:", data.shape)
print(data.head())

X = data.drop(columns=["class"])
y = data["class"]

le = LabelEncoder()
y = le.fit_transform(y)

print("Features:", X.shape, "Target:", y.shape)

for col in X.columns:
    X[col] = X[col].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape, "Validation set:", X_val.shape)

clf = xgb.XGBClassifier(
    n_estimators=200,
    random_state=42,
    enable_categorical=True,
    tree_method="hist",
    use_label_encoder=False,
    eval_metric="logloss"
)

clf.fit(X_train, y_train)

import joblib
cat_info = {col: X[col].cat.categories.tolist() for col in X.columns}

joblib.dump(clf, "mushroom_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(cat_info, "mushroom_categories.pkl")

y_val_pred = clf.predict(X_val)
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
