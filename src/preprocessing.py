import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

def load_data(file_path: str) -> pd.DataFrame:
    """
    Lee un archivo CSV o Excel y lo devuelve como un DataFrame de pandas.
    Lanza una excepción si el archivo no existe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    #Determinar si es CSV o Excel
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and Excel files are supported")

    return df

def check_target_variable (df: pd.DataFrame, target_variable: str):
    """
    Verifica que la variable objetivo exista en el DataFrame.
    Lanza ValueError si no existe.
    """
    if target_variable not in df.columns:
        raise ValueError(f'Target variable "{target_varaible}" not found in DataFrame')
    return True

def clean_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Limpia valores nulos según la estrategia indicada:
    - 'drop': elimina filas con nulos.
    - 'mean': imputa en numéricos con la media, y en categóricas con la moda.
    - 'median': imputa en numéricos con la mediana, y en categóricas con la moda.
    """
    if strategy == "drop":
        df = df.dropna()
    elif strategy in ("mean", "median"):
        for col in df.select_dtypes(include=[np.number]).columns:
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        #Para columnas categóricas o tipo object, imputamos con la moda
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        pass
    return df

def remove_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Elimina outliers de las columnas numéricas usando la regla IQR.
    factor=1.5 suele ser un estándar, pero se puede ajustar.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        #Filtar el DataFrame en ese rango
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas u 'object' en numéricas.
    - Si la columna tiene solo 2 categorías, se aplica LabelEncoder.
    - Si tiene más, se aplica One-Hot Encoding.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if df[col].nunique() == 2:
            #Label Encoding binario
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            #One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df

def scale_data(df: pd.DataFrame,
               scaler_type: str = "standard",
               ignore_target: str = None) -> pd.DataFrame:
    """
    Escala las columnas numéricas para homogeneizar rangos.
    - scaler_type = 'standard' usa StandardScaler.
    - scaler_type = 'minmax' usa MinMaxScaler.
    - ignore_target = nombre de la columna que NO se debe escalar.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    #Evitar escalar la variable objectivo si se indica.
    if ignore_target in numeric_cols:
        numeric_cols.remove(ignore_target)

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        #Si no coincide con lo esperado, devolvemos el df tal cual.
        return df
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
