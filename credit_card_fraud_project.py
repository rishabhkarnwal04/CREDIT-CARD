import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv('creditcard.csv')

data.info()

data.head()

data.describe()

data.isnull().sum()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Class', palette='Set2')
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Time', y='Amount', hue='Class', palette='Set1')
plt.title('Time vs Transaction Amount')
plt.xlabel('Time (seconds)')
plt.ylabel('Transaction Amount')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Class', y='Amount', palette='Set3')
plt.title('Transaction Amount by Class (Fraud vs Non-Fraud)')
plt.xlabel('Class')
plt.ylabel('Transaction Amount')
plt.show()

data['Class'].value_counts()

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

print(legit.shape)
print(fraud.shape)

legit.Amount.describe()

fraud.Amount.describe()

data.groupby('Class').mean()

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset.head()

new_dataset.describe()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
X

Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)

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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

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

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc(model, X_test, Y_test, name):
    fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]):.2f})')

plt.figure(figsize=(10, 6))
for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(f"\n{name}")
    print(classification_report(Y_test, Y_pred))
    plot_conf_matrix(Y_test, Y_pred, name)
    plot_roc(model, X_test, Y_test, name)

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split, GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='f1', cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, Y_train)

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
Y_pred = best_model.predict(X_test)
print("\nTuned Random Forest")
print(classification_report(Y_test, Y_pred))
plot_conf_matrix(Y_test, Y_pred, "Tuned Random Forest")

import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")

    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    legit_sample = legit.sample(n=len(fraud), random_state=42)
    new_data = pd.concat([legit_sample, fraud])

    X = new_data.drop(columns='Class')
    y = new_data['Class']

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = load_data()

model = LGBMClassifier()
model.fit(X_train, y_train)

st.title("💳 Credit Card Fraud Detection (LightGBM)")
st.markdown("Enter values for the features below to check if a transaction is fraudulent.")

user_input = {}
selected_features = ['V1', 'V2', 'V3', 'V4', 'V14', 'V10', 'V11', 'Amount']

for feature in selected_features:
    user_input[feature] = st.slider(f"{feature}", float(X_train[feature].min()), float(X_train[feature].max()), 0.0)

input_df = pd.DataFrame([user_input])

for feature in X_train.columns:
    if feature not in input_df.columns:
        input_df[feature] = 0.0

input_df = input_df[X_train.columns]


if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("This is likely a **fraudulent** transaction.")
    else:
        st.success(" This is likely a **legitimate** transaction.")


