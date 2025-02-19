# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
from typing import Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(
        "./files/input/train_data.csv.zip", index_col=False, compression="zip"
    )
    test_data = pd.read_csv(
        "./files/input/test_data.csv.zip", index_col=False, compression="zip"
    )
    return train_data, test_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.rename(columns={"default payment next month": "default"})
    df_cleaned = df_cleaned.drop(columns=["ID"])
    df_cleaned = df_cleaned.loc[df_cleaned["MARRIAGE"] != 0]
    df_cleaned = df_cleaned.loc[df_cleaned["EDUCATION"] != 0]
    df_cleaned["EDUCATION"] = df_cleaned["EDUCATION"].apply(
        lambda x: 4 if x >= 4 else x
    )
    df_cleaned = df_cleaned.dropna()
    return df_cleaned


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["default"]), df["default"]


def create_pipeline() -> Pipeline:
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(), categorical_features)],
        remainder=MinMaxScaler(),
    )
    feature_selector = SelectKBest(score_func=f_regression, k=10)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection", feature_selector),
            ("classifier", classifier),
        ]
    )


def create_estimator(pipeline: Pipeline, x_train: pd.DataFrame) -> GridSearchCV:
    param_grid = {
        "feature_selection__k": range(1, len(x_train.columns) + 1),
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }
    return GridSearchCV(
        pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1, refit=True
    )


def save_model(path: str, estimator: GridSearchCV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def calculate_metrics(
    dataset_type: str, y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, Any]:
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion_matrix(
    dataset_type: str, y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def save_metrics(metrics: list[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")


def main():
    train_data, test_data = load_data()
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    x_train, y_train = split_data(train_data)
    x_test, y_test = split_data(test_data)

    pipeline = create_pipeline()
    estimator = create_estimator(pipeline, x_train)
    estimator.fit(x_train, y_train)

    save_model("files/models/model.pkl.gz", estimator)

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    train_metrics = calculate_metrics("train", y_train, y_train_pred)
    test_metrics = calculate_metrics("test", y_test, y_test_pred)

    train_cm = calculate_confusion_matrix("train", y_train, y_train_pred)
    test_cm = calculate_confusion_matrix("test", y_test, y_test_pred)

    save_metrics(
        [train_metrics, test_metrics, train_cm, test_cm], "files/output/metrics.json"
    )


if __name__ == "__main__":
    main()
