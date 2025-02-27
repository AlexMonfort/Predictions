# models/trainer.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def train_supervised_model(model,
                           X,
                           y,
                           problem_type: str,
                           batch_size: int = 32,
                           epochs: int = 50,
                           lr: float = 0.001,
                           patience: int = 5):
    """
    Entrena un modelo PyTorch para clasificación o regresión.
    Devuelve:
    - model: el modelo PyTorch entrenado
    - history: un dict con la evolución de la pérdida (loss)
    
    Parámetros:
    - model: instancia de NeuralNetwork (PyTorch)
    - X, y: datos (numpy arrays)
    - problem_type: "classification" o "regression"
    - batch_size, epochs, lr, patience: hiperparámetros básicos
      (patience se ignora en este ejemplo, no implementamos EarlyStopping).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convertir a tensores
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    # Para clasificación multiclase se requiere y como long (entero)
    if problem_type == "classification":
        unique_classes = len(set(y))
        if unique_classes > 2:
            y_t = torch.tensor(y, dtype=torch.long).to(device)
        else:
            # Binario => Tensor float (para BCELoss)
            y_t = torch.tensor(y, dtype=torch.float32).to(device)
    else:
        # Regresión => float
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

    # Definir la pérdida
    if problem_type == "classification":
        if unique_classes > 2:
            # Multiclase => CrossEntropy
            criterion = nn.CrossEntropyLoss()
        else:
            # Binario => BCELoss (tenemos salida sigmoide)
            criterion = nn.BCELoss()
    else:
        # Regresión => MSE
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": []}  # Podrías meter "val_loss" si implementas validación

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            if problem_type == "classification" and unique_classes > 2:
                # multiclase => outputs (batch_size, num_classes), batch_y (batch_size)
                loss = criterion(outputs, batch_y)
            elif problem_type == "classification":
                # binario => outputs (batch_size,1), batch_y (batch_size)
                loss = criterion(outputs.squeeze(), batch_y)
            else:
                # regresión => outputs (batch_size,1), batch_y (batch_size)
                loss = criterion(outputs.squeeze(), batch_y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        history["loss"].append(epoch_loss)

        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    return model, history
