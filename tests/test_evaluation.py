# tests/test_evaluation.py

import numpy as np
from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_clustering,
    evaluate_model
)

def test_evaluate_classification():
    y_true = [0,1,1,0,1]
    y_pred = [0,1,0,0,1]
    metrics = evaluate_classification(y_true, y_pred, average='binary', plot_confusion=False)
    assert "accuracy" in metrics
    assert "f1_score" in metrics

def test_evaluate_regression():
    y_true = [2.5, 0.0, 2.1, 7.8]
    y_pred = [3.0, -0.1, 2.0, 7.9]
    metrics = evaluate_regression(y_true, y_pred)
    assert "mse" in metrics
    assert "r2" in metrics

def test_evaluate_clustering():
    X = np.array([[1,2],[2,2],[10,10],[9,10]])
    labels = [0,0,1,1]
    metrics = evaluate_clustering(X, labels)
    assert "silhouette" in metrics

def test_evaluate_model_classification():
    y_true = [0,0,1,1]
    y_pred = [0,1,1,1]
    metrics = evaluate_model(
        problem_type='classification',
        y_true=y_true,
        y_pred=y_pred
    )
    assert "accuracy" in metrics

def test_evaluate_model_regression():
    y_true = [10,20,30]
    y_pred = [12,18,31]
    metrics = evaluate_model(
        problem_type='regression',
        y_true=y_true,
        y_pred=y_pred
    )
    assert "r2" in metrics

def test_evaluate_model_clustering():
    X = np.array([[0,1],[0,2],[10,10],[9,9]])
    labels = [0,0,1,1]
    metrics = evaluate_model(
        problem_type='clustering',
        X=X,
        labels=labels
    )
    assert "silhouette" in metrics
