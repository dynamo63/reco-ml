import streamlit as st
import numpy as np
import joblib

# Presentation
st.title(
    """
        Bienvenue sur reco-ml
    """
)

st.subheader("Un modele de machine learning pour predire la valeur d'un bit sur 4 features")

# Creation de la fonction de prediction
@st.cache
def predict_bit_value(values: list) -> int:
    model = joblib.load('reco.joblib')
    features = np.array(values).reshape(-1, 4)
    return model.predict(features)[0]

# Creation des inputs
total_bits = st.number_input("Nombre de Bits", value=0)
number_bit_1 = st.number_input("Nombre de bit a 1", value=0)
level = st.number_input("Rang dans la sous famille", value=0)
bit_posit = st.number_input("Position du bit", value=0)

btn_predict_value = st.button("Predire la valeur du bit")

if btn_predict_value:
    # Obtention des features
    data = [total_bits, number_bit_1, level, bit_posit]
    # prediction
    prediction = predict_bit_value(data)

    st.header(f"La valeur possible de ce bit est : {prediction}")
    
