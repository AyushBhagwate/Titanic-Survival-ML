# 🚢 Titanic Survival Prediction (Logistic Regression)

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using a **Logistic Regression model**.
It demonstrates a complete Machine Learning pipeline including **EDA, data preprocessing, model training, and visualization**.

---

## 📂 Dataset

* Source: Kaggle Titanic Dataset
* Target Variable: **Survived**

  * `0` → Did not survive
  * `1` → Survived

---

## 🧠 Problem Type

* **Classification Problem**

---

## 🔍 Exploratory Data Analysis (EDA)

Performed:

* Data overview (`shape`, `info`, `describe`)
* Missing value analysis
* Numerical feature analysis
* Categorical feature analysis
* Correlation analysis
* Target variable distribution (Survived vs Not Survived)

---

## 📊 Outputs

Generated inside `outputs/` folder:

* `metrics.txt` → Model performance
* `confusion_matrix.png` → Classification results
* `roc_curve.png` → Model performance curve
* `predictions.csv` → Actual vs Predicted values

---

## 📁 Project Structure


project/
│
├── data/
│   └── titanic.csv
│
├── models/
│   └── titanic_model.pkl
│
├── outputs/
│   ├── metrics.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── predictions.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md


---

## 🚀 How to Run

### 1. Clone the repository

- git clone <your-repo-link>
- cd <project-folder>

### 2. Install dependencies

- pip install -r requirements.txt

### 3. Run the project

- python main.py

---

## 📈 Key Insights

* Logistic Regression performs well for binary classification
* Features like **Sex, Pclass, Fare, Age** significantly influence survival
* ROC-AUC helps evaluate model performance beyond accuracy

---

## 📌 Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Hyperparameter tuning
* Handle class imbalance
* Deploy using Streamlit

---

## 👨‍💻 Author
Ayush Bhagwate
