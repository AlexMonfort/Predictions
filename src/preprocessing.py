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
        raise ValueError(f'Target variable "{target_variable}" not found in DataFrame')
    return True

def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Genera un reporte básico de calidad de datos
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicated_rows": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # En MB
    }
    return report

def process_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    except Exception as e:
        print(f"Warning: Error processing date columns: {str(e)}")
    return df
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
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())
        # Para columnas categóricas o tipo object, imputamos con la moda
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
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
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        #Filtar el DataFrame en ese rango
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas u 'object' en numéricas.
    - Si la columna tiene solo 2 categorías, se aplica LabelEncoder.
    - Si tiene más, se aplica One-Hot Encoding.
    """
    df = df.copy()  # Create a copy to avoid pandas warnings
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if df[col].nunique() == 2:
            # Label Encoding binario
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            # One-Hot Encoding with proper prefix and column names
            dummies = pd.get_dummies(df[col], prefix=col)
            # Drop the original column and join the dummy columns
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
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

def feature_selection_random_forest (df: pd.DataFrame,
                                     target_variable: str,
                                     problem_type: str,
                                     threshold: float = 0.01) -> pd.DataFrame:
    """
    Usa un RandomForest bàsico para medir la importancia de las variables y se queda sólo con
    aquellas que superen 'threshold' de importancia.
    - problem_type = 'classification' o 'regression'.
    """
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)

    importances = model.feature_importances_
    feat_names = np.array(X.columns)

    # Seleccionamos las features relevantes
    selected_feats = feat_names[importances >= threshold]
    df_reduced = df[selected_feats.tolist() + [target_variable]]

    return df_reduced

def unsupervised_feature_reduction(df: pd.DataFrame,
                                   var_threshold: float = 0.0,
                                   corr_threshold: float = 0.95) -> pd.DataFrame:
    """
    Para el caso no supervisado:
    1) Elimina features con varianza muy baja (VarianceThreshold).
    2) Elimina features muy correlacionadas (> corr_threshold).
    Devuelve un DataFrame reducido.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # 1) VarianceThreshold
    if var_threshold > 0:
        vt = VarianceThreshold(threshold=var_threshold)
        vt.fit(numeric_df)
        kept_cols = numeric_df.columns[vt.get_support()]
        numeric_df = numeric_df[kept_cols]

    # 2) Eliminar correlaciones altas
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    col_to_drop = [c for c in upper_triangle.columns if any(upper_triangle[c] > corr_threshold)]
    numeric_df = numeric_df.drop(columns=col_to_drop, errors='ignore')

    # Unimos con las columnas no numéricas (que ya podrían haberse codificado)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    reduced_df = pd.concat([numeric_df, df[non_numeric_cols]], axis=1)

    return reduced_df
def preprocess_data(file_path: str,
                   target_variable: str = None,
                   clean_strategy: str = "mean",
                   remove_outliers: bool = True,
                   outlier_factor: float = 1.5,
                   scaler_type: str = "standard",
                   supervised_threshold: float = 0.01,
                   var_threshold: float = 0.0,
                   corr_threshold: float = 0.95) -> pd.DataFrame:
    # 1. Cargar DataFrame
    df = load_data(file_path)
    
    # 1.1 Validación de calidad (nuevo)
    quality_report = validate_data_quality(df)
    print("Data Quality Report:", quality_report)
    
    # 1.2 Procesamiento de fechas (nuevo)
    df = process_date_columns(df)
    
    # 2. Validar si es supervisado
    is_supervised = False
    problem_type = None
    if target_variable:
        check_target_variable(df, target_variable)
        is_supervised = True
        # Determinamos clasificación o regresión según el tipo de datos en la columna objetivo
        if pd.api.types.is_numeric_dtype(df[target_variable]):
            problem_type = "regression"
        else:
            problem_type = "classification"

    # 3. Limpieza de valores nulos
    df = clean_missing_values(df, strategy=clean_strategy)

    # 4. Outliers (opcional)
    if remove_outliers and is_supervised:
        df = remove_outliers_iqr(df, factor=outlier_factor)

    # 5. Codificación de categóricas
    df = encode_categorical(df)

    # 6. Escalado (evitar escalar la variable objetivo si es numérica en regresión)
    ignore_target_col = target_variable if (is_supervised and problem_type == "regression") else None
    df = scale_data(df, scaler_type=scaler_type, ignore_target=ignore_target_col)

    # 7. Selección/reducción de variables
    if is_supervised:
        df = feature_selection_random_forest(df, target_variable, problem_type, threshold=supervised_threshold)
    else:
        df = unsupervised_feature_reduction(df, var_threshold=var_threshold, corr_threshold=corr_threshold)

    return df