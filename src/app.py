import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---

# Ubicamos la raíz del proyecto (Path(__file__)) y subimos dos niveles con --> .parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'lgbm_car_pricing_model.joblib'
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.joblib"
MAPPING_PATH = BASE_DIR / 'models' / 'vehicle_mapping.joblib'

# --- CARGA DE ACTIVOS ---


@st.cache_resource  # Esto evita que el modelo se cargue en cada click, optimizando la App
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    mapping = joblib.load(MAPPING_PATH)

    return model, columns, mapping


model, model_columns, mapping_logic = load_model_assets()

# --- INTERFAZ ---

st.set_page_config(page_title='RUSTY BARGAIN | PRICING', layout='wide')

st.title('TASADOR DE AUTOS RUSTY BARGAIN')

# 1. Extraemos las marcas de los vehículos (list comprehension)
# k[0] es la marca en la tupla --> (marca, vechiculo)
lista_marcas = sorted(
    list(
        set(
            [k[0] for k in mapping_logic.keys()]
        )
    )
)

with st.container(border=True):

    # Seteamos las dos columnas que contendrán nuestros inputs
    col1, col2 = st.columns(2)

    with col1:

        st.markdown('## VEHÍCULO')

        # A) Selección de la marca
        brand_selected = st.selectbox(
            label='Marca',
            options=lista_marcas,
            index=0
        )

        # B) Selección del modelo (Encadenado a la marca). Filtramos las keys
        # que coincidan con la brand_selected
        modelos_disponibles = sorted(
            k[1] for k in mapping_logic.keys() if k[0] == brand_selected
        )

        model_selected = st.selectbox(
            label='Modelo',
            options=modelos_disponibles
        )

        # C) Selección de la carrocería (encadenado a la marca y el modelo)
        # Accedemos a la lista de valores usando marca-modelo
        carroceria_disponible = mapping_logic.get(
            (brand_selected, model_selected)
        )

        body_selected = st.segmented_control(
            label='Tipo de carrocería',
            options=carroceria_disponible
        )

        # D) Transmisión
        gearbox_selected = st.pills(
            label='Transmisión',
            options=['auto', 'manual']
        )

        # E) Estado
        repaired_selected = st.radio(
            label='¿Reparado?',
            options=['yes', 'no', 'unknown']
        )

    with col2:

        st.markdown('## CARACTERÍSTICAS')

        # --- Inputs numéricos ---

        # A) Año de registro
        reg_year = st.number_input(
            label='Año de registro',
            min_value=1950,
            max_value=2016,
            value=2010
        )

        # B) Mes de registro: definimos diccionario MES:valor
        # extraemos los nombres para el selectbox
        month_map = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
                     'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
                     'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}

        month_name = st.selectbox(
            label='Mes de registro',
            options=list(month_map.keys()),
            index=0
        )

        # C) Potencia
        power = st.number_input(
            label='Potencia [HP]',
            min_value=10,
            max_value=600,
            value=150
        )

        # D) Kilometraje
        mileage = st.number_input(
            label='Kilometraje',
            min_value=5000,
            max_value=150000,
            step=5000
        )

        # --- Inputs categóricos ---

        # E) Combustible
        fuel_selected = st.segmented_control(
            label='Tipo de combustible',
            options=['petrol', 'gasoline', 'lpg',
                     'cng', 'hybrid', 'electric', 'other']
        )

st.markdown('---')

# --- LÓGICA DE PREDICCIÓN ---

# Botón de acción
predict_btn = st.button(
    label='CALCULAR VALOR DE MERCADO',
    use_container_width=True
)

# Comprobación rápida
if predict_btn == True:

    # A) Verificación de campos vacíos
    if (not body_selected) or (not fuel_selected) or (not gearbox_selected):
        st.error('¡HAY CAMPOS VACÍOS! Termínalos, animal.')

    # B) Mapeo del mes
    reg_month_numeric = month_map.get(month_name)

    # C) Creación de diccionario (usar nombres exactos que el modelo espera)
    input_data = {
        'VehicleType': body_selected,
        'RegistrationYear': reg_year,
        'Gearbox': gearbox_selected,
        'Power': power,
        'Model': model_selected,
        'Mileage': mileage,
        'RegistrationMonth': reg_month_numeric,
        'FuelType': fuel_selected,
        'Brand': brand_selected,
        'NotRepaired': repaired_selected
    }

    # D) COnversión a DataFrame y Reordenamiento
    df_input = pd.DataFrame(data=[input_data])
    df_input = df_input[model_columns]

    # E) Casting de categorías
    cat_features = ['VehicleType', 'Gearbox', 'Gearbox',
                    'Model', 'FuelType', 'Brand', 'NotRepaired']

    for col in cat_features:
        df_input[col] = df_input[col].astype(dtype='category')

    # F) Predicción final
    prediction = model.predict(X=df_input)[0]

# --- PRESENTACIÓN DE RESULTADOS ---

    st.markdown('---')

    st.header('RREPORTE DE VALORACIÓN')

    # Definimos el error del modelo (obtenido DEL notebook)
    MODEL_RMSE = 1539.43

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(
            label="Precio Estimado",
            value=f"€ {prediction:,.2f}",
            delta=f"Error +/- €{MODEL_RMSE:,.0f}",
            delta_color="off"
        )

    with col_res2:
        lower_bound = max(0, prediction - MODEL_RMSE)
        upper_bound = prediction + MODEL_RMSE

        st.write(f"### Rango de confianza")
        st.write(
            f"Basado en patrones históricos, el valor real del vehículo oscila entre:")
        st.info(f"**€ {lower_bound:,.2f} — € {upper_bound:,.2f}**")

    # Mensaje de valor agregado
    if reg_year < 2000:
        st.warning("Nota de Clásico: Al ser un vehículo de más de 25 años, el precio puede variar significativamente por el estado de conservación mecánica.")
    elif mileage > 120000:
        st.info("Impacto de Kilometraje: El alto kilometraje es el factor principal de depreciación en esta valoración.")
