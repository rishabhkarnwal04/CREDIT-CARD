import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

data = pd.read_csv('creditcard.csv')

print(data.info())


print(data['Class'].value_counts())

legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=42)
new_dataset = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2, stratify=Y, random_state=42)

def evaluate_model(model, name):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("-" * 60)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    evaluate_model(model, name)

try:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    evaluate_model(xgb_model, "XGBoost")
except ImportError:
    print("XGBoost not installed. Skipping XGBoost model.")

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title("💳 Credit Card Fraud Detection")

# Sidebar info
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

