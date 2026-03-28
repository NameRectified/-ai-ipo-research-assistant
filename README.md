# IPO Profitability Prediction System

## Overview
Predicts whether an IPO will be profitable at listing using subscription data.

The final model focuses on key investor demand signals (QIB and RII) and is optimized for high-precision decision-making.

---

## Features
- Predicts probability of IPO profitability  
- Compared multiple models (Logistic Regression, Random Forest, XGBoost, Neural Networks)  
- Performed feature engineering using subscription ratios  
- Applied L1 regularization for feature selection  
- Optimized decision threshold for high precision (~0.93)  
- Deployed as an interactive Streamlit app  

---

## Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn  
- Streamlit  

---

## Model Insights
- Institutional subscription (QIB) is the strongest predictor  
- Retail subscription (RII) provides additional signal  
- Many features (price, size, HNI, ratios) were removed via L1 regularization  

---

## Model Performance
- Final Features: Subscription_QIB, Subscription_RII  
- ROC-AUC: ~0.75  
- Precision: ~0.93 (at threshold = 0.55)  
- Recall: ~0.43  

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo
[Live App](https://ipo-profitability-prediction.streamlit.app/)

## Additional Resources
Training notebook: 
[Colab file](https://colab.research.google.com/drive/1FHCvRh4MbRyZJf3eDlinFiuPvLWxd1B5?usp=sharing)