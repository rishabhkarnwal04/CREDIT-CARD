**Credit Card Fraud Detection System**

---

**Project Name:**
Credit Card Fraud Detection using Machine Learning

**By:**
Rishabh Karnwal

---
This project implements a machine learning-based credit card fraud detection system using a dataset of transactions provided in the `creditcard.csv` file. The system aims to accurately identify fraudulent transactions from legitimate ones using classification algorithms.
link for csv file --> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
---

**Dataset:**

* **File:** `creditcard.csv`
* **Source:** Contains anonymized transaction data from European cardholders over a two-day period in September 2013.
* **Size:** 284,807 transactions with 492 fraudulent cases.
* **Features:**

  * 28 anonymized principal components (`V1` to `V28`)
  * `Time`: Seconds elapsed between each transaction and the first transaction
  * `Amount`: Transaction amount
  * `Class`: Target variable (0 = normal, 1 = fraud)

---

**Technologies Used:**

* Python 3.x
* Libraries:

  * pandas
  * numpy
  * matplotlib / seaborn (for visualization)
  * scikit-learn (for machine learning models and evaluation)

---

**Steps Involved:**

1. **Data Preprocessing:**

   * Handling class imbalance using undersampling/oversampling
   * Feature scaling (StandardScaler/MinMaxScaler)
2. **Model Selection:**

   * Algorithms used: Logistic Regression, Random Forest, Decision Tree, etc.
   * Cross-validation for performance evaluation
3. **Model Evaluation:**

   * Accuracy, Precision, Recall, F1 Score
   * Confusion Matrix and ROC-AUC Curve
4. **Result Analysis:**

   * Comparative analysis of different models
   * Insights into most effective algorithms for fraud detection

---

**How to Run:**

1. Ensure Python and required libraries are installed.
2. Download `creditcard.csv` file
3. Run the main script 
4. Observe model performance and visual outputs.

---

**Future Improvements:**

* Real-time detection implementation using a live API
* Use of deep learning techniques (e.g., Autoencoders, LSTMs)
* Integration with banking software for alerts and automation

---

