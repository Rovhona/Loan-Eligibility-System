import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from datetime import datetime

st.set_page_config(page_title="Loan Eligibility Dashboard", layout="wide")

st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #f5f7fa; }
        h1, h2, h3 { color: #003262; text-align: center; }
        .stButton>button { background-color: #003262; color: white; font-weight: bold; border-radius: 8px; }
        .stDownloadButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("loan_eligibility_model_logistic_regression.pkl")

model_package = load_model()

class LoanEligibilityPredictor:
    def __init__(self, model, scaler, label_encoders, feature_columns, model_name="Model"):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        self.model_name = model_name

    def preprocess(self, applicant_data):
        df = pd.DataFrame([applicant_data])
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Missing")
                df[col] = le.transform(df[col])

        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["LoanAmountToIncome"] = df["LoanAmount"] / (df["TotalIncome"] + 1)
        df["IncomePerDependent"] = df["ApplicantIncome"] / (df["Dependents"].replace(0, 1))
        df["ApplicantIncome_log"] = np.log(df["ApplicantIncome"] + 1)
        df["TotalIncome_log"] = np.log(df["TotalIncome"] + 1)
        df["LoanAmount_log"] = np.log(df["LoanAmount"] + 1)

        df = df.reindex(columns=self.feature_columns)
        df = pd.DataFrame(self.scaler.transform(df), columns=self.feature_columns)
        return df

    def predict(self, applicant_data):
        try:
            X_processed = self.preprocess(applicant_data)
            prediction = self.model.predict(X_processed)[0]
            proba = self.model.predict_proba(X_processed)[0]
            return {
                "Loan_ID": applicant_data.get("Loan_ID", "Unknown"),
                "status": "✅ Approved" if prediction == 1 else "❌ Rejected",
                "probability_approved": round(proba[1], 3),
                "probability_rejected": round(proba[0], 3)
            }
        except Exception as e:
            error_msg = str(e)
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()} | Loan_ID: {applicant_data.get('Loan_ID', 'Unknown')} | Error: {error_msg}\n")
            return {
                "Loan_ID": applicant_data.get("Loan_ID", "Unknown"),
                "status": "⚠️ Prediction Failed",
                "probability_approved": None,
                "probability_rejected": None,
                "error": error_msg
            }

def predict_batch(df, model_package):
    predictor = LoanEligibilityPredictor(
        model=model_package['model'],
        scaler=model_package['scaler'],
        label_encoders=model_package['label_encoders'],
        feature_columns=model_package['feature_columns'],
        model_name=model_package['model_name']
    )
    return pd.DataFrame([predictor.predict(row.to_dict()) for _, row in df.iterrows()])

def export_results_csv(results_df):
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv, "loan_results.csv", "text/csv")

# ---------------- UI ----------------
st.markdown("<h1 style='text-align: center;'>🏦 Loan Eligibility Prediction</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Batch Upload", "🧍 Single Prediction"])

with tab1:
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} with {len(input_df)} rows")
        st.dataframe(input_df.head())

        with st.spinner("🔍 Making predictions..."):
            results_df = predict_batch(input_df, model_package)

        def highlight_status(row):
            if row["status"] == "✅ Approved":
                return ['background-color: #d4edda'] * len(row)
            elif row["status"] == "❌ Rejected":
                return ['background-color: #f8d7da'] * len(row)
            return [''] * len(row)

        st.subheader("📋 Prediction Results")
        st.dataframe(results_df.style.apply(highlight_status, axis=1))

        export_results_csv(results_df)

        st.subheader("📊 Visual Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Loan Approval Distribution")
            status_counts = results_df["status"].value_counts()
            fig1, ax1 = plt.subplots()
            status_counts.plot(kind='bar', color=['green', 'red'], ax=ax1)
            ax1.set_ylabel("Number of Applicants")
            st.pyplot(fig1)

        with col2:
            st.markdown("#### Approval Confidence")
            if "probability_approved" in results_df.columns:
                st.line_chart(results_df[["probability_approved", "probability_rejected"]])

        with st.expander("📈 Model Insights"):
            if hasattr(model_package['model'], 'coef_'):
                importances = model_package["model"].coef_[0]
                feature_importance = pd.Series(importances, index=model_package["feature_columns"])
                top_features = feature_importance.abs().sort_values(ascending=False).head(10)
                st.bar_chart(top_features)

        with st.expander("📊 Real-Time Stats"):
            approved = results_df[results_df["status"] == "✅ Approved"]
            total = len(results_df)
            if total > 0:
                st.metric("Approval Rate", f"{len(approved) / total * 100:.2f}%")
                st.metric("Avg Approval Confidence", f"{approved['probability_approved'].mean():.2f}")

    else:
        st.info("👈 Please upload a CSV file to start prediction.")

with tab2:
    st.subheader("🧍 Predict for a Single Applicant")
    with st.form("single_applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            loan_id = st.text_input("Loan ID")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", [0, 1, 2, 3])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_term = st.selectbox("Loan Term", [360, 180, 120, 60])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

        submit = st.form_submit_button("Predict")

    if submit:
        single_data = {
            "Loan_ID": loan_id,
            "Gender": gender,
            "Married": married,
            "Dependents": int(dependents),
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }

        result = LoanEligibilityPredictor(
            model=model_package['model'],
            scaler=model_package['scaler'],
            label_encoders=model_package['label_encoders'],
            feature_columns=model_package['feature_columns']
        ).predict(single_data)

        st.subheader("🧾 Prediction Result")
        st.json(result)

# ---------------- Footer & Model Info ----------------
with st.expander("ℹ️ Model Details"):
    st.write(f"**Model**: {model_package['model_name']}")
    st.write(f"**Version**: {model_package.get('model_version', 'N/A')}")
    st.write(f"**Training Date**: {model_package.get('training_date', 'Unknown')}")
    st.metric("Accuracy", f"{model_package.get('accuracy', 'N/A')}")
    st.metric("Precision", f"{model_package.get('precision', 'N/A')}")
    st.metric("Recall", f"{model_package.get('recall', 'N/A')}")

st.markdown("""
---
<p style='text-align: center; font-size: 14px;'>
    Model by Rovhona Mudau
</p>
""", unsafe_allow_html=True)
