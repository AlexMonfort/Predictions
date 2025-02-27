"""
evaluation.py

Contiene funciones para evaluar modelos de:
1) Clasificación (binaria o multiclase)
2) Regresión
3) Clustering

Además, incluye una función 'evaluate_model' que decide automáticamente
qué métricas calcular según el tipo de problema.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score
)
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, average='binary', plot_confusion=False, labels=None):
    """
    Calcula métricas de clasificación (accuracy, precision, recall, f1).
    Param:
    - y_true: array de etiquetas verdaderas
    - y_pred: array de etiquetas predichas
    - average: 'binary', 'macro', 'micro', 'weighted' (según multiclase o binario)
    - plot_confusion: si True, muestra la matriz de confusión
    - labels: listado de labels si deseas personalizar el orden en la confusión
    Retorna un diccionario con las métricas calculadas.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    metrics_dict = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

    if plot_confusion:
        # Matriz de confusión y su despliegue
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión")
        plt.show()

    return metrics_dict


def evaluate_regression(y_true, y_pred):
    """
    Calcula métricas de regresión clásicas: MSE, RMSE y R².
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    metrics_dict = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    return metrics_dict


def evaluate_clustering(X, labels):
    """
    Calcula la métrica silhouette para evaluar la calidad de clusters.
    Si sólo hay 1 cluster, silhouette no se puede calcular y se retorna None.
    Param:
    - X: matriz de features (numpy array o DataFrame)
    - labels: array con las asignaciones de cada fila a un cluster
    Retorna un diccionario con 'silhouette'.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        sil_score = silhouette_score(X, labels)
    else:
        sil_score = None  # No se puede calcular silhouette con un solo cluster

    metrics_dict = {
        'silhouette': sil_score
    }
    return metrics_dict


def evaluate_model(problem_type: str,
                   y_true=None,
                   y_pred=None,
                   X=None,
                   labels=None,
                   average='binary',
                   plot_confusion=False):
    """
    Función unificada para evaluar un modelo según el tipo de problema.
    Param:
    - problem_type: 'classification', 'regression' o 'clustering'
    - y_true, y_pred: utilizados en clasificación y regresión
    - X, labels: utilizados en clustering
    - average: criterio de agregación para métricas en clasificación ('binary', 'macro', etc.)
    - plot_confusion: si True, se muestra la matriz de confusión en clasificación
    Retorna un diccionario con las métricas calculadas o None si no procede.
    """
    if problem_type == 'classification':
        if y_true is None or y_pred is None:
            raise ValueError("Para clasificación se requieren y_true e y_pred.")
        metrics_dict = evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            average=average,
            plot_confusion=plot_confusion
        )
        return metrics_dict

    elif problem_type == 'regression':
        if y_true is None or y_pred is None:
            raise ValueError("Para regresión se requieren y_true e y_pred.")
        metrics_dict = evaluate_regression(y_true, y_pred)
        return metrics_dict

    elif problem_type == 'clustering':
        if X is None or labels is None:
            raise ValueError("Para clustering se requieren X y labels.")
        metrics_dict = evaluate_clustering(X, labels)
        return metrics_dict

    else:
        raise ValueError("El parámetro 'problem_type' debe ser 'classification', 'regression' o 'clustering'.")
