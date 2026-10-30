import pandas as pd

COLUMNAS_MODELO = ['yr', 'mnth', 'hr', 'workingday', 'weathersit', 'temp', 'windspeed']
COLUMNAS_CATEGORICAS = ['weathersit']

# Cargar el CSV
def load_data(path):
    return pd.read_csv(path)


#Quita las columnas que contienen (o suman) el target, y los identificadores.
def drop_leakage(df):
    return df.drop(columns=['casual', 'registered', 'instant', 'dteday'])


#Se queda con las columnas del modelo más el target cnt."""
def select_features(df, columns=COLUMNAS_MODELO):
    return df[columns + ['cnt']]


#Convierte las columnas categóricas en dummies."""
def encode_dummies(df, columns=COLUMNAS_CATEGORICAS):
    return pd.get_dummies(df, columns=columns)

# Separa features y target.
def split_xy(df):
    features = df.drop(columns='cnt')
    target = df['cnt']
    return features, target


# aplica todos los pasos y devuelve X e y
def clean_data(path):
    df = load_data(path)
    df = drop_leakage(df)
    df = select_features(df)
    df = encode_dummies(df)
    return split_xy(df)