# IPO Profitability Prediction System

## Overview
Predicts whether an IPO will be profitable at listing using subscription and pricing data.

---

## Features
- Predicts probability of IPO profitability  
- Compares multiple ML models (Logistic Regression, Random Forest, XGBoost, NN)  
- Optimized decision threshold for better precision  
- Interactive Streamlit app for real-time predictions  

---

## Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn  
- Streamlit  

---

## Model Performance
- Best Model: Logistic Regression  
- ROC-AUC: 0.71  
- Precision improved via threshold tuning  

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py