import streamlit as st
import pandas as pd
import joblib
from pathlib ipmort Path

# --- CONFIGURACIÓN DE RUTAS ---

# Ubicamos la raíz del proyecto (Path(__file__)) y subimos dos niveles con --> .parent.parent
BASE_DIR = Path(__file__).resolve.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'lgbm_car_pricing_model.joblib'
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.joblib"

# --- CARGA DE ACTIVOS ---


@st.cache_resource  # Esto evita que el modelo se cargue en cada click, optimizando la App
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)

    return model, columns


model, model_columns = load_model_assets()

# --- INTERFAZ ---
st.title('TASADOR DE AUTOS RUSTY BARGAIN')
st.write('"Calcula el precio de mercado de forma instantánea usando Machine Learning (LightGBM Model).')
