# tests/test_preprocessing.py

import os
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    load_data,
    validate_data_quality,
    process_date_columns,
    clean_missing_values,
    remove_outliers_iqr,
    encode_categorical,
    scale_data,
    preprocess_data
)

def test_load_data_csv(tmp_path):
    # Creamos un CSV de prueba
    data = {"col1": [1,2,3], "col2": [4,5,6]}
    df_test = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df_test.to_csv(file_path, index=False)

    df_loaded = load_data(str(file_path))
    assert not df_loaded.empty
    assert list(df_loaded.columns) == ["col1", "col2"]

def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("no_exist.csv")

def test_validate_data_quality():
    df = pd.DataFrame({
        "A": [1,2,3],
        "B": [4,5,6],
        "C": [None, None, 9]
    })
    report = validate_data_quality(df)
    assert report["total_rows"] == 3
    assert report["total_columns"] == 3
    assert report["missing_values"]["C"] == 2
    # No falla => passed

def test_process_date_columns():
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2020-01-01", "2020-02-02", "2020-03-03"])
    })
    df_processed = process_date_columns(df)
    # Verifica que se hayan creado las columnas fecha_year, fecha_month, etc.
    assert "fecha_year" in df_processed.columns
    assert "fecha_month" in df_processed.columns
    assert df_processed["fecha_year"].iloc[0] == 2020

def test_clean_missing_values():
    df = pd.DataFrame({
        "num": [1, 2, np.nan],
        "cat": ["a", "b", None]
    })
    df_cleaned = clean_missing_values(df, strategy="mean")
    # num -> se imputa con la media (1.5), cat -> se imputa con la moda ("a" o "b")
    assert df_cleaned["num"].isnull().sum() == 0
    assert df_cleaned["cat"].isnull().sum() == 0

def test_remove_outliers_iqr():
    df = pd.DataFrame({
        "x": [10, 12, 15, 1000, 13, 14],  # 1000 = outlier
        "y": [1,1,1,1,1,1]
    })
    df_clean = remove_outliers_iqr(df, threshold=1.5)
    # Esperamos que la fila con 1000 se elimine
    assert len(df_clean) == 5
    assert 1000 not in df_clean["x"].values

def test_encode_categorical():
    df = pd.DataFrame({
        "bin_cat": ["yes", "no", "yes"],
        # 3 categorías: apple, banana, orange
        "multi_cat": ["apple", "banana", "orange"]
    })
    df_encoded = encode_categorical(df)
    
    # 'bin_cat' => 2 categorías => LabelEncoder => sigue llamándose 'bin_cat'
    assert "bin_cat" in df_encoded.columns

    # 'multi_cat' => 3 categorías => One-Hot => con drop_first => 2 columnas
    # Posibles columnas: multi_cat_banana, multi_cat_orange
    assert "multi_cat_banana" in df_encoded.columns
    assert "multi_cat_orange" in df_encoded.columns

def test_scale_data():
    df = pd.DataFrame({
        "num1": [10, 20, 30],
        "num2": [100, 200, 300],
        "target": [0,1,0]
    })
    # Escalamos, ignorando la variable objetivo
    df_scaled = scale_data(df, scaler_type="standard", ignore_target="target")
    # Revisamos que "target" no se haya alterado y que los demás se escalaron
    assert df_scaled["target"].equals(df["target"])
    # Comprobamos que num1 y num2 tengan media 0
    mean_num1 = np.round(df_scaled["num1"].mean(), 5)
    mean_num2 = np.round(df_scaled["num2"].mean(), 5)
    assert abs(mean_num1) < 1e-6  # ~ 0
    assert abs(mean_num2) < 1e-6  # ~ 0

def test_preprocess_data(tmp_path):
    # Integra varios pasos (carga, limpieza, codificación, escalado, etc.)
    data = {
        "f1": [10, 12, None, 15],
        "cat": ["A", "B", "B", "A"],
        "target": [0, 1, 1, 0]
    }
    df_test = pd.DataFrame(data)
    file_csv = tmp_path / "temp_data.csv"
    df_test.to_csv(file_csv, index=False)

    df_processed = preprocess_data(
        file_path=str(file_csv),
        target_variable="target",
        clean_strategy="mean",
        remove_outliers=False,  # no outliers for now
        scaler_type="standard",
        supervised_threshold=0.0  # para no filtrar variables
    )
    # Comprobamos que el DF resultante no esté vacío
    assert not df_processed.empty
    assert "target" in df_processed.columns
