# tests/test_models.py
import pytest
import numpy as np

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Skip all TensorFlow-dependent tests if TensorFlow is not available
pytestmark = pytest.mark.skipif(
    not TENSORFLOW_AVAILABLE,
    reason="TensorFlow is not installed or not compatible with current Python version"
)
def test_build_neural_network_classification():
    # Construimos un modelo binario
    model = build_neural_network(
        problem_type="classification",
        input_dim=5,
        hidden_layers=[8,4],
        num_classes=1,
        learning_rate=0.001
    )
    assert model is not None
    # Verificamos la capa de salida
    assert model.layers[-1].output_shape == (None, 1)

def test_build_neural_network_regression():
    model = build_neural_network(
        problem_type="regression",
        input_dim=3,
        hidden_layers=[10],
        num_classes=1
    )
    # Última capa lineal
    assert model.layers[-1].output_shape == (None, 1)
    # Revisamos la pérdida configurada (MSE)
    assert model.loss == "mse"

def test_cluster_data():
    X = np.array([
        [1,2],
        [1,3],
        [10,20],
        [11,19]
    ])
    labels = cluster_data(X, n_clusters=2)
    # Verificamos que se crean 2 clusters
    assert len(np.unique(labels)) == 2

def test_find_optimal_clusters():
    X = np.random.rand(10,2)
    best_k, inertias = find_optimal_clusters(X, max_k=5)
    assert 1 <= best_k <= 5
    assert len(inertias) == 5

def test_train_supervised_model():
    from tensorflow.keras import Sequential
    
    # Modelo Keras dummy
    model = Sequential()
    model.add(
        # capa de entrada con 4 neuronas
        # y luego 1 neurona de salida (binaria)
    )
    # Para simplificar, construimos un minired neuronal:
    model.add(
        build_neural_network(
            problem_type="classification",
            input_dim=4,
            hidden_layers=[4],
            num_classes=1
        ).layers[-1]  # reusamos la capa final
    )
    
    # Iniciamos con compile manual (o reusamos el build_neural_network entero)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    X = np.random.randn(20,4)
    y = np.random.randint(0,2, size=(20,))
    
    trained_model, history = train_supervised_model(
        model, X, y, validation_split=0.2, epochs=5, batch_size=5, patience=2
    )
    assert trained_model is not None
    # Comprobamos que se entrenó
    assert len(history.history['loss']) > 0

def test_run_model_classification():
    # Simulamos un dataset de clasificación binaria con 2 features
    X = np.array([[1,1],[2,2],[3,3],[4,4]])
    y = np.array([0,0,1,1])

    model, metrics_dict, history = run_model(
        X=X, y=y, problem_type="classification",
        hidden_layers=[4],
        epochs=2,  # entrenamos poco
        batch_size=2
    )
    # Esperamos ver accuracy en metrics_dict
    assert "accuracy" in metrics_dict

def test_run_model_regression():
    X = np.array([[1],[2],[3],[4]])
    y = np.array([2.0,4.0,6.0,8.0])  # y = 2x
    
    model, metrics_dict, history = run_model(
        X=X, y=y, problem_type="regression",
        hidden_layers=[4],
        epochs=2,
        batch_size=2
    )
    # Esperamos ver 'r2' en metrics_dict
    assert "r2" in metrics_dict

def test_run_model_clustering():
    X = np.array([
        [0,0],[0,1],[10,10],[10,11]
    ])
    labels, metrics_dict = run_model(X=X, y=None, n_clusters=2)
    # 2 clusters => check silhouette
    assert "silhouette" in metrics_dict
