# 💊 Medical Insurance Charges Predictor - [	👀 Try It Out!](https://medical-insurance-predictorr.streamlit.app/)

A simple, intuitive web app built with **Streamlit** that predicts the estimated **medical insurance cost** based on user input like age, BMI, number of children, region, and smoking status.

This project uses a **Lasso Regression model** trained on the popular [`insurance.csv`](https://www.kaggle.com/datasets/mirichoi0218/insurance) dataset and demonstrates how machine learning can be used for real-world regression problems.

---

## 🚀 Features

- Predicts medical charges using:
  - Age
  - Gender
  - BMI
  - Number of children
  - Smoking status
  - Region
- Trained using **Lasso Regression** with log-transformed target
- One-hot encoding and feature scaling for preprocessing
- User-friendly web interface with real-time prediction

---

## 🧠 How It Works

1. The user inputs personal and lifestyle information.
2. The app encodes and scales this data using a saved `scaler.pkl`.
3. The trained Lasso model (`model.pkl`) predicts the **log of charges**.
4. The prediction is reverse-transformed using `np.expm1()` to give actual cost.

---
---

## 💻 Run It Locally

## 🛠️ Setup Instructions

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   
3. **Run the app**  
   ```bash
   streamlit run app.py
   
## 📊 Dataset

This app is trained on the [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance), which contains:

- 1,300+ records of individuals
- Demographic information
- Health-related attributes
- Corresponding medical charges

---

## ⚙️ Model Info

- **Algorithm:** Lasso Regression  
- **Target Variable:** `log1p(charges)` (log-transformed for stability)  
- **Performance:**
  - 📈 **R² Score (log scale):** ~0.74  
  - 📉 **RMSE (₹ scale):** ~₹2,000–₹3,000  

---