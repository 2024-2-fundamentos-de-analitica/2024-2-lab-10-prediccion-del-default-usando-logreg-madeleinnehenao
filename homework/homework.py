import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import json


# Paso 1: Cargar los datos y limpieza
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = (
        4  # Agrupar valores superiores de EDUCATION en 'others'
    )
    return df


df_train = load_and_clean_data(
    "C:\\Users\ASUS\Desktop\\2024-2-lab-10-prediccion-del-default-usando-logreg-madeleinnehenao\\files\\input\\train_data.csv.zip"
)
df_test = load_and_clean_data(
    "C:\\Users\ASUS\Desktop\\2024-2-lab-10-prediccion-del-default-usando-logreg-madeleinnehenao\\files\\input\\test_data.csv.zip"
)

# Paso 2: División en X e Y
X_train = df_train.drop(columns=["default"])
y_train = df_train["default"]
X_test = df_test.drop(columns=["default"])
y_test = df_test["default"]

# Paso 3: Crear pipeline
pipeline = Pipeline(
    [
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("scaler", MaxAbsScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
        ("model", LogisticRegression()),
    ]
)

# Paso 4: Optimización de hiperparámetros
param_grid = {"feature_selection__k": [5, 10, 15], "model__C": [0.1, 1, 10]}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring="balanced_accuracy")
grid_search.fit(X_train, y_train)

# Guardar el mejor modelo
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)


# Paso 6: Cálculo de métricas
def compute_metrics(y_true, y_pred, dataset_type):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

metrics = [
    compute_metrics(y_train, y_train_pred, "train"),
    compute_metrics(y_test, y_test_pred, "test"),
]


# Paso 7: Cálculo de matrices de confusión
def compute_confusion_matrix(y_true, y_pred, dataset_type):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        "true_1": {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]},
    }


metrics.append(compute_confusion_matrix(y_train, y_train_pred, "train"))
metrics.append(compute_confusion_matrix(y_test, y_test_pred, "test"))

# Guardar métricas
with open("files/output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
