# tests/test_integration.py

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data
from src.models.model_factory import run_model

def test_full_pipeline_supervised(tmp_path):
    # Construimos un CSV con datos supervisados (clasificación).
    data = {
        "f1": [1,2,3,4],
        "f2": [10,9,8,7],
        "target": [0,0,1,1]
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_supervised.csv"
    df.to_csv(csv_file, index=False)

    df_preprocessed = preprocess_data(
        file_path=str(csv_file),
        target_variable="target",
        clean_strategy="mean",
        remove_outliers=False,  # dataset chico
        scaler_type="standard"
    )
    # Separamos X e y
    y = df_preprocessed["target"].values
    X = df_preprocessed.drop(columns=["target"]).values

    model, metrics, history = run_model(X, y, problem_type="classification", epochs=2, batch_size=2)
    
    assert "accuracy" in metrics  # ya indica que pudo entrenar

def test_full_pipeline_unsupervised(tmp_path):
    # Dataset sin target -> clustering
    data = {
        "f1": [0,1,10,11],
        "f2": [1,2,10,12]
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_unsupervised.csv"
    df.to_csv(csv_file, index=False)

    df_preprocessed = preprocess_data(
        file_path=str(csv_file),
        target_variable=None,  # no supervisado
        var_threshold=0.0,
        corr_threshold=0.95
    )
    # Llamar run_model con y=None
    X = df_preprocessed.values
    labels, metrics = run_model(X=X, y=None, n_clusters=2)
    assert "silhouette" in metrics
