# models/model_factory.py

import numpy as np
import torch

from models.neural_network import build_neural_network
from models.clustering import cluster_data
from models.trainer import train_supervised_model
from models.metrics import classification_metrics, regression_metrics, clustering_metrics

def process_supervised(X,
                       y,
                       problem_type: str,
                       hidden_layers=[64, 32],
                       epochs=50,
                       batch_size=32,
                       lr=0.001,
                       patience=5):
    """
    Crea y entrena un modelo PyTorch (clasificación o regresión).
    Devuelve (trained_model, metrics_dict, history)
    """
    # Detectar si es binario (num_classes=1) o multiclase
    if problem_type == "classification":
        unique_labels = np.unique(y)
        if len(unique_labels) > 2:
            num_classes = len(unique_labels)  # multiclase
        else:
            num_classes = 1  # binario
    else:
        num_classes = 1  # regresión

    # Construir la red
    input_dim = X.shape[1]
    model = build_neural_network(
        problem_type=problem_type,
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        num_classes=num_classes
    )

    # Entrenar
    trained_model, history = train_supervised_model(
        model=model,
        X=X,
        y=y,
        problem_type=problem_type,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience
    )

    # Generar predicciones en CPU
    trained_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = trained_model(X_t).numpy()  # float
        # Si classification multiclase => shape (N, num_classes)
        # Si binaria => shape (N, 1)
        # Si regression => shape (N, 1)

    # Convertir preds a labels si es clasificación
    if problem_type == "classification":
        if num_classes == 1:
            # binario => threshold=0.5
            y_pred = (preds >= 0.5).astype(int).ravel()
        else:
            y_pred = np.argmax(preds, axis=1)
        # Calcular métricas
        metrics_dict = classification_metrics(y, y_pred, average='binary' if num_classes==1 else 'macro')
    else:
        # regresión
        y_pred = preds.ravel()
        metrics_dict = regression_metrics(y, y_pred)

    return trained_model, metrics_dict, history


def process_unsupervised(X, n_clusters=3, random_state=42):
    """
    Aplica K-means (scikit-learn) o la técnica de clustering que uses en clustering.py
    Devuelve (labels, metrics_dict).
    """
    labels = cluster_data(X, n_clusters=n_clusters, random_state=random_state)
    metrics_dict = clustering_metrics(X, labels)
    return labels, metrics_dict


def run_model(X,
              y=None,
              problem_type=None,
              hidden_layers=[64, 32],
              epochs=50,
              batch_size=32,
              lr=0.001,
              patience=5,
              n_clusters=3,
              random_state=42):
    """
    Función unificada:
    - Si y=None => no supervisado (clustering).
    - Si y != None => supervisado (detectar o usar 'problem_type').
    """
    if y is None:
        # No supervisado
        labels, metrics_dict = process_unsupervised(X, n_clusters, random_state)
        return labels, metrics_dict

    # Supervisado
    if problem_type is None:
        # Deducir si es numerica => regression, si no => classification
        if np.issubdtype(y.dtype, np.number):
            # Podría ser binaria o multiclase. Lo resolvemos en process_supervised
            # con la cuenta de clases. De momento asumimos 'classification' si int, 'regression' si float...
            # O lo definimos con un if
            # Para simplificar:
            # - si hay más de 5 valores distintos => reg
            # - si < 5 => clas
            unique_vals = np.unique(y)
            if len(unique_vals) > 5:
                problem_type = "regression"
            else:
                problem_type = "classification"
        else:
            problem_type = "classification"

    trained_model, metrics_dict, history = process_supervised(
        X, y,
        problem_type=problem_type,
        hidden_layers=hidden_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience
    )
    return trained_model, metrics_dict, history
