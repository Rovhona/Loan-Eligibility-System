# 🏦 Loan Eligibility Prediction Dashboard

A modern and interactive dashboard for predicting loan eligibility using machine learning, built with **Streamlit**.

![dashboard-preview](https://github.com/Rovhona/Loan-Eligibility-System/blob/main/assets/page1.png) 
![dashboard-preview](https://github.com/Rovhona/Loan-Eligibility-System/blob/main/assets/page2.png) 


---

## ✅ Project Status

> 🚀 **First working version is live!**  
> We've implemented core features and but still there prediction errors. Ongoing improvements include user experience, export features, and enhanced model performance.

---

## 📆 Dataset Used

We used the public dataset from Kaggle:

🔗 [Loan CSV Dataset](https://www.kaggle.com/datasets/tanishaj225/loancsv/data)

This dataset contains attributes like:

- Gender, Married, Education, Self_Employed  
- Applicant & Coapplicant Income  
- Loan Amount, Loan Term, Credit History  
- Property Area  
- Loan Status (Target)

---

## 🚀 Key Features

- 📁 Upload CSVs with applicant data  
- 🧠 Predict loan approval with a trained ML model  
- ⚙️ Handle both bulk predictions and single applicant input  
- 📊 View prediction confidence and data distribution  
- 📥 Download results as CSV or PDF  
- 📋 View model accuracy, precision, and recall on dashboard  
- 💅 Modern and interactive UI with Streamlit  

---

## 🧠 Model Overview

We trained multiple models (Logistic Regression, Random Forest, Gradient Boosting) and selected **Logistic Regression** for production due to balanced performance.

Model components saved include:
- Trained model  
- Scaler  
- Label encoders  
- Feature column order  
- Performance metrics  

Full training and evaluation in:

📓 [`Loan_Eligibility_System.ipynb`](Loan_Eligibility_System.ipynb)

---

## 🔧 Fixes and Enhancements

- ✅ **Prediction Error Fixed**: Ensured feature names at inference match training pipeline  
- ✅ Integrated a consistent preprocessing pipeline for real-time predictions  
- ✅ Improved error handling with detailed messages  
- 📊 Added live model evaluation metrics  
- 💾 Export predictions as PDF (with charts and summary)  

---

## 💡 Planned Improvements

- 🎯 Train with XGBoost or LightGBM for better performance  
- 🎨 Use advanced charts with Plotly or Altair  
- 🔐 Add user authentication (Supabase/Auth0)  
- 🐳 Add Dockerfile for containerized deployment  
- ☁️ Deploy on Streamlit Cloud or Hugging Face  
- 📉 Add error logging to file/database  

---

## 🛠️ Setup Instructions

Clone the repo and run locally:

```bash
git clone https://github.com/Rovhona/Loan-Eligibility-System.git
cd Loan-Eligibility-System
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Sample Input for Testing

Use this sample for single prediction testing:

```json
{
  "Loan_ID": "LP001234",
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": 1,
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
```



## 🙇‍♂️ Author

**Rovhona Mudau**  
🔗 [GitHub Profile](https://github.com/Rovhona)

---

## ✍️ Contributing

We welcome contributions!

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Submit a Pull Request

---

## 👅 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---
