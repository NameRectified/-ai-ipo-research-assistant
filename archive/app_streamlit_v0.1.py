import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.title("IPO Profitability Predictor")
st.write("This model predicts IPO profitability using institutional (QIB) and retail (RII) subscription data.")
st.write("Enter IPO details:")


qib = st.number_input("QIB Subscription", min_value=0.0)
rii = st.number_input("RII Subscription", min_value=0.0)

THRESHOLD = 0.55
if st.button("Predict"):
    if qib == 0 and rii == 0:
        st.warning("Please enter valid subscription values.")
    else:
        features = np.array([[qib, rii]])
        features_scaled = scaler.transform(features)

        prob = model.predict_proba(features_scaled)[0][1]

        st.subheader(f"Probability of Profit: {prob:.2f}")

        if prob > THRESHOLD:
            st.success("Likely Profitable IPO")
        else:
            st.error("Risky IPO")

# Links section
st.markdown("---")
with st.expander("Project Details"):
    st.write("For implementation details and training process:")
    st.markdown("[Github repository](https://github.com/NameRectified/IPO-Prediction)")
    st.markdown("[Training Notebook (Colab)](https://colab.research.google.com/drive/1FHCvRh4MbRyZJf3eDlinFiuPvLWxd1B5?usp=sharing)")