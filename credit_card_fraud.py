import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

st.title("💳 Credit Card Fraud Detection")

st.sidebar.header("Transaction Input")
st.sidebar.markdown("Enter transaction details below:")

X_demo, y_demo = make_classification(n_samples=1000, n_features=29, random_state=42)
scaler = StandardScaler()
X_demo_scaled = scaler.fit_transform(X_demo)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_demo_scaled, y_demo)

user_input = []
for i in range(1, 29):
    val = st.sidebar.number_input(f"V{i}", value=0.0)
    user_input.append(val)

amount = st.sidebar.number_input("Transaction Amount", value=0.0)
user_input.append(amount)

user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)


if st.button("🔍 Predict Fraud"):
    prediction = model.predict(user_df_scaled)[0]
    probability = model.predict_proba(user_df_scaled)[0][1]
    if prediction == 1:
        st.error(f"🚨 Alert! This transaction is predicted to be fraudulent with {probability*100:.2f}% confidence.")
    else:
        st.success(f"✅ This transaction appears to be legitimate. Confidence: {(1-probability)*100:.2f}%")


st.markdown("---")
st.caption("This demo uses synthetic data and a Random Forest model trained on generated features.")

