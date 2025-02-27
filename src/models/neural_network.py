# models/neural_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    """
    Construye una red neuronal secuencial para clasificación o regresión.
    - Si problem_type='classification' y num_classes=1 => binario (salida sigmoide).
    - Si problem_type='classification' y num_classes>1 => multiclase (salida lineal + se usa CrossEntropy).
    - Si problem_type='regression' => salida lineal (1 neurona).
    """
    def __init__(self, input_dim, hidden_layers, problem_type="classification", num_classes=1):
        super(NeuralNetwork, self).__init__()
        self.problem_type = problem_type
        self.num_classes = num_classes

        layers = []
        in_features = input_dim

        # Capas ocultas
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        # Capa de salida
        if problem_type == "classification":
            # Binaria => 1 neurona, multiclase => num_classes neuronas
            out_features = num_classes if num_classes > 1 else 1
            layers.append(nn.Linear(in_features, out_features))
        else:
            # Regresión => 1 neurona lineal
            layers.append(nn.Linear(in_features, 1))

        # Creamos el Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Define la pasada hacia delante.
        - Para binario se aplica Sigmoid a la salida (1 neurona).
        - Para multiclase, se asume que la pérdida CrossEntropy se encargará
          de la softmax interna (no se aplica softmax aquí).
        - Para regresión, salida lineal.
        """
        out = self.model(x)
        if self.problem_type == "classification" and self.num_classes == 1:
            # Binaria => aplicar sigmoide
            out = torch.sigmoid(out)
        return out


def build_neural_network(problem_type: str,
                         input_dim: int,
                         hidden_layers: list = [64, 32],
                         num_classes: int = 1):
    """
    Construye la clase NeuralNetwork de PyTorch.
    (Equivalente a la función build_model de Keras, pero sin compilar, 
    porque en PyTorch se define la optimización en la fase de entrenamiento).
    """
    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        problem_type=problem_type,
        num_classes=num_classes
    )
    return model
