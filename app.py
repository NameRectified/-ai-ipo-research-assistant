import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("IPO Profitability Predictor")
st.write("Enter IPO details:")

# Inputs
qib = st.number_input("QIB Subscription", min_value=0.0)
hni = st.number_input("HNI Subscription", min_value=0.0)
rii = st.number_input("RII Subscription", min_value=0.0)
price = st.number_input("Issue Price", min_value=0.0)
size = st.number_input("Issue Size", min_value=0.0)

# Prediction
if st.button("Predict"):
    if qib == 0 and hni == 0 and rii == 0:
        st.warning("Please enter valid subscription values.")
    else:
        features = np.array([[qib, hni, rii, price, size]])
        features_scaled = scaler.transform(features)

        prob = model.predict_proba(features_scaled)[0][1]

        st.subheader(f"Probability of Profit: {prob:.2f}")

        if prob > 0.5:
            st.success("Likely Profitable IPO")
        else:
            st.error("Risky IPO")

# Links section
st.markdown("---")
with st.expander("Project Details"):
    st.write("For implementation details and training process:")
    st.markdown("[Training Notebook (Colab)](https://colab.research.google.com/drive/1FHCvRh4MbRyZJf3eDlinFiuPvLWxd1B5?usp=sharing)")