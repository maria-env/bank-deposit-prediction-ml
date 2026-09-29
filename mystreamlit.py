import streamlit as st
import pandas as pd
import joblib


# Configuración base
st.set_page_config(page_title="Predictor Bancario", layout="wide")

st.title("Sistema de Predicción de Marketing Bancario")
st.markdown("Introduce los datos del cliente para saber si contratará el depósito.")

# Cargamos el modelo
@st.cache_resource
def load_model():
    try:
        return joblib.load('modelo_final.joblib')
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

if model is not None:
    # Definimos el aside para introducir los datos
    st.sidebar.header("Datos del Cliente")
    
    def user_input_features():
        # Tomamos los valores de las features numéricas
        age = st.sidebar.slider("Edad", 18, 95, 35)
        balance = st.sidebar.number_input("Balance (€)", value=1000)
        day = st.sidebar.slider("Día del mes", 1, 31, 15)
        duration = st.sidebar.number_input("Duración llamada (seg)", value=200)
        campaign = st.sidebar.slider("Contactos campaña actual", 1, 50, 1)
        pdays = st.sidebar.number_input("Días desde último contacto (-1 si nunca)", value=-1)
        previous = st.sidebar.number_input("Contactos previos", value=0)

        # y de las features categóricas
        job = st.sidebar.selectbox("Trabajo", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.sidebar.selectbox("Estado Civil", ['married', 'single', 'divorced'])
        education = st.sidebar.selectbox("Educación", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.sidebar.selectbox("¿Impagos?", ['no', 'yes'])
        housing = st.sidebar.selectbox("¿Hipoteca?", ['no', 'yes'])
        loan = st.sidebar.selectbox("¿Préstamo?", ['no', 'yes'])
        contact = st.sidebar.selectbox("Contacto", ['cellular', 'telephone', 'unknown'])
        month = st.sidebar.selectbox("Mes", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        poutcome = st.sidebar.selectbox("Resultado anterior", ['success', 'failure', 'other', 'unknown'])

        data = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
            'contact': contact, 'day': day, 'month': month, 'duration': duration,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
        }

        return pd.DataFrame([data])

    input_df = user_input_features()

    st.subheader("Datos a evaluar")
    st.write(input_df)

    # 4. BOTÓN DE PREDICCIÓN
    if st.button("Calcular Predicción"):
        try:
            # Crea la columna generada con feature engineering
            input_df['contacted_before'] = (input_df['pdays'] != -1).astype(int)
            
            cols_que_espera = model.feature_names_in_
            input_df = input_df[cols_que_espera]
            
            # Predicción
            prediction = model.predict(input_df)
            
            st.divider()
            
            # Resultado visual
            resultado = str(prediction[0]).lower()
            
            if resultado in ['1', 'yes']:
                st.success("### EL CLIENTE CONTRATARÁ EL DEPÓSITO")
                st.balloons()
            else:
                st.error("### EL CLIENTE NO CONTRATARÁ EL DEPÓSITO")
                
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)
                st.info(f"Probabilidad de éxito: {prob[0][1]*100:.2f}%")
                
        except Exception as e:
            st.error(f"Hubo un error al procesar los datos: {e}")
            st.info("Revisa si el modelo espera columnas que no están en el formulario.")

else:
    st.warning("No se pudo cargar el archivo 'modelo_final.joblib'. Asegúrate de que esté en la misma carpeta que este script.")