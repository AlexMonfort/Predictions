# main.py

import argparse
import numpy as np
import pandas as pd

# Preprocesamiento (ya adaptado a tu gusto)
from preprocessing import preprocess_data

# Importa la lógica unificada para entrenar/clustering
from models.model_factory import run_model

# (Opcional) Para métricas extras o graficar confusiones
from evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Software de Predicción con PyTorch")

    # Argumentos para preprocesamiento
    parser.add_argument("--data_path", type=str, required=True, help="CSV/Excel con los datos.")
    parser.add_argument("--target_variable", type=str, default=None, help="Variable objetivo (si hay).")

    parser.add_argument("--clean_strategy", type=str, default="mean", help="drop/mean/median")
    parser.add_argument("--remove_outliers", action="store_true", help="Si se incluye, elimina outliers (solo supervisado).")
    parser.add_argument("--outlier_factor", type=float, default=1.5, help="Umbral IQR.")
    parser.add_argument("--scaler_type", type=str, default="standard", help="standard/minmax")
    parser.add_argument("--supervised_threshold", type=float, default=0.01, help="Umbral RF para descartar features.")
    parser.add_argument("--var_threshold", type=float, default=0.0, help="Umbral varianza (no supervisado).")
    parser.add_argument("--corr_threshold", type=float, default=0.95, help="Umbral correlación (no supervisado).")

    # Argumentos para el modelado
    parser.add_argument("--hidden_layers", type=str, default="64,32", help="Capas, p.ej. '64,32'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_clusters", type=int, default=3)

    # Ejemplo: para graficar matriz confusión
    parser.add_argument("--plot_confusion", action="store_true",
                        help="Si se incluye, grafica la matriz de confusión (clasificación).")

    args = parser.parse_args()

    # 1) Preprocesar
    df = preprocess_data(
        file_path=args.data_path,
        target_variable=args.target_variable,
        clean_strategy=args.clean_strategy,
        remove_outliers=args.remove_outliers,
        outlier_factor=args.outlier_factor,
        scaler_type=args.scaler_type,
        supervised_threshold=args.supervised_threshold,
        var_threshold=args.var_threshold,
        corr_threshold=args.corr_threshold
    )

    # 2) Supervisado o clustering
    if args.target_variable:
        print(f"[INFO] Modo supervisado. Target = {args.target_variable}")
        y = df[args.target_variable].values
        X = df.drop(columns=[args.target_variable]).values

        # Parsear hidden_layers
        hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",") if x.strip().isdigit()]

        model_or_labels, metrics_dict, history = run_model(
            X=X, y=y,
            hidden_layers=hidden_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            patience=args.patience
        )
        print("[RESULTADOS SUPERVISADO]", metrics_dict)

        # (Opcional) Ver si es clasificación binaria => graficar confusión
        if "accuracy" in metrics_dict and args.plot_confusion:
            # Obtenemos predicciones
            # model_or_labels es 'trained_model' en supervisado
            # Haz una eval rápida
            pass
            # ... (Puedes llamar evaluate_model y graficar)

    else:
        print("[INFO] Modo no supervisado (clustering).")
        X = df.values
        labels, metrics_dict = run_model(
            X=X, y=None,
            n_clusters=args.n_clusters
        )
        print("[RESULTADOS CLUSTERING]", metrics_dict)
        print("Labels:", labels)

    print("[INFO] Proceso completado.")

if __name__ == "__main__":
    main()
