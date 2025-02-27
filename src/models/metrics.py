from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score,
    silhouette_score
)

def classification_metrics(y_true, y_pred, average='binary'):
    """
    Calcula métricas de clasificación (accuracy, precision, recall, f1).
    average='binary' para problemas binarios,
    'macro' o 'micro' para multiclase.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }


def regression_metrics(y_true, y_pred):
    """
    Calcula métricas de regresión (MSE, RMSE, R2).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def clustering_metrics(X, labels):
    """
    Calcula la métrica silhouette para evaluar la calidad de los clusters.
    Cuanto más cercano a 1, mejor; valores cerca de 0 indican clusters solapados.
    """
    if len(set(labels)) > 1:  # Al menos 2 clusters
        score = silhouette_score(X, labels)
    else:
        score = None  # No se puede calcular silhouette si sólo hay 1 cluster
    return {
        'silhouette': score
    }
