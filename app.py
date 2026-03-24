import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("IPO Profitability Predictor")
st.write("Enter ipo details:")

qib = st.number_input("QIB Subscription", min_value=0.0)
hni = st.number_input("HNI Subscription", min_value=0.0)
rii = st.number_input("RII Subscription", min_value=0.0)
price = st.number_input("Issue Price", min_value=0.0)
size = st.number_input("Issue Size", min_value=0.0)

if st.button("Predict"):
    features = np.array([[qib, hni, rii, price, size]])

    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    st.subheader(f"Probability of profit: {prob:.2f}")

    if prob > 0.5:
        st.success("Likely profitable IPO")
    else:
        st.error("Risky ipo")