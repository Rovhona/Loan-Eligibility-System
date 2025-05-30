import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------
# 🚀 LoanEligibilityPredictor Class
# -------------------------------
class LoanEligibilityPredictor:
    def __init__(self, model, scaler, label_encoders, feature_columns, model_name="Model"):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        self.model_name = model_name

    def preprocess(self, applicant_data):
        df = pd.DataFrame(applicant_data)

        # Label Encoding
        for col, le in self.label_encoders.items():
            if col in df.columns:
                valid_classes = le.classes_
                df[col] = df[col].apply(lambda x: x if x in valid_classes else valid_classes[0])
                df[col + '_encoded'] = le.transform(df[col].astype(str))

        # Feature Engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanAmountToIncome'] = df['LoanAmount'] / df['TotalIncome']
        df['Dependents_num'] = df['Dependents'].replace('3+', 3).astype(float)
        df['IncomePerDependent'] = df['TotalIncome'] / (df['Dependents_num'] + 1)
        df['ApplicantIncome_log'] = np.log1p(df['ApplicantIncome'])
        df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
        df['LoanAmount_log'] = np.log1p(df['LoanAmount'])

        # Reorder Columns
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Scale Numerical Columns
        num_cols = [
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'TotalIncome', 'LoanAmountToIncome', 'IncomePerDependent',
            'ApplicantIncome_log', 'TotalIncome_log', 'LoanAmount_log'
        ]
        indices = [self.feature_columns.index(col) for col in num_cols]
        df_scaled = df.copy()
        df_scaled.iloc[:, indices] = self.scaler.transform(df.iloc[:, indices])

        return df_scaled

    def predict(self, applicant_data):
        try:
            X_processed = self.preprocess(applicant_data)
            predictions = self.model.predict(X_processed)
            probas = self.model.predict_proba(X_processed)

            results = []
            for i, (pred, proba) in enumerate(zip(predictions, probas)):
                results.append({
                    "Loan_ID": applicant_data.iloc[i].get("Loan_ID", "Unknown"),
                    "status": "✅ Approved" if pred == 1 else "❌ Rejected",
                    "probability_approved": round(proba[1] * 100, 2),
                    "probability_rejected": round(proba[0] * 100, 2)
                })

            return results
        except Exception as e:
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()} | Prediction | Error: {str(e)}\n")

            return [{
                "Loan_ID": applicant_data.iloc[0].get("Loan_ID", "Unknown") if not applicant_data.empty else "Unknown",
                "status": "⚠️ Prediction Failed",
                "probability_approved": None,
                "probability_rejected": None,
                "error": str(e)
            }]

# -------------------------------
# 📦 Load Model
# -------------------------------
try:
    model_package = joblib.load("loan_eligibility_model_logistic_regression.pkl")
    predictor = LoanEligibilityPredictor(
        model=model_package['model'],
        scaler=model_package['scaler'],
        label_encoders=model_package['label_encoders'],
        feature_columns=model_package['feature_columns'],
        model_name=model_package['model_name']
    )
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# -------------------------------
# 🎯 Streamlit UI
# -------------------------------
st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")
st.title("🏦 Loan Eligibility Prediction App")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📤 Batch Upload", "🧍 Single Prediction"])

# -------------------------------
# 📤 Batch Upload Tab
# -------------------------------
with tab1:
    st.header("📤 Batch Prediction Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="batch_uploader")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {uploaded_file.name} with {len(df)} rows")

            required_cols = [
                'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Property_Area'
            ]
            filtered_df = df.dropna(subset=required_cols).copy()
            dropped_rows = len(df) - len(filtered_df)

            st.info(f"✔️ Cleaned {dropped_rows} incomplete rows")

            test_samples = filtered_df
            results = predictor.predict(test_samples)

            # Result Table
            table_data = []
            for i, (idx, row) in enumerate(test_samples.iterrows()):
                table_data.append({
                    'Sample': idx,
                    'Loan_ID': results[i]['Loan_ID'],
                    'Gender': row['Gender'],
                    'Married': row['Married'],
                    'Dependents': row['Dependents'],
                    'Education': row['Education'],
                    'Self_Employed': row['Self_Employed'],
                    'ApplicantIncome': row['ApplicantIncome'],
                    'CoapplicantIncome': row['CoapplicantIncome'],
                    'LoanAmount': row['LoanAmount'],
                    'Loan_Amount_Term': row['Loan_Amount_Term'],
                    'Credit_History': row['Credit_History'],
                    'Property_Area': row['Property_Area'],
                    'Status': results[i]['status'],
                    'Probability_Approved (%)': results[i]['probability_approved'],
                    'Probability_Rejected (%)': results[i]['probability_rejected']
                })

            st.subheader("📋 Prediction Results")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # Stats
            approved_count = sum(r['status'] == "✅ Approved" for r in results)
            approval_rate = (approved_count / len(results)) * 100 if results else 0
            avg_approval_conf = np.mean([r['probability_approved'] for r in results if r['probability_approved'] is not None])

            st.subheader("📊 Summary Stats")
            col1, col2 = st.columns(2)
            col1.metric("Approval Rate", f"{approval_rate:.2f}%")
            col2.metric("Avg Approval Confidence", f"{avg_approval_conf:.2f}%")

            # 📈 Visualization
            st.subheader("📊 Visual Insights")
            fig1, ax1 = plt.subplots()
            ax1.bar(["Approved", "Rejected"], [approved_count, len(results) - approved_count], color=["#4CAF50", "#F44336"])
            ax1.set_title("Loan Approval Distribution")
            ax1.set_ylabel("Applications")
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots()
            ax2.hist(
                [r['probability_approved'] for r in results if r['probability_approved'] is not None],
                bins=20, color="#2196F3", edgecolor='black'
            )
            ax2.set_title("Approval Confidence Distribution")
            ax2.set_xlabel("Confidence (%)")
            ax2.set_xlim(0, 100)
            st.pyplot(fig2)
            plt.close(fig2)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()} | File Upload | Error: {str(e)}\n")

# -------------------------------
# 🧍 Single Prediction Tab
# -------------------------------
with tab2:
    st.header("🧍 Single Application Prediction")
    st.write("Provide the following details:")

    with st.form(key="single_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            loan_id = st.text_input("Loan ID (optional)", value="Unknown")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0, step=1.0)
            loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0.0, value=360.0, step=12.0)
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submit_button = st.form_submit_button(label="🔍 Predict Eligibility")

        if submit_button:
            input_data = {
                "Loan_ID": loan_id,
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }

            input_df = pd.DataFrame([input_data])
            result = predictor.predict(input_df)[0]

            st.subheader("📋 Prediction Result")
            with st.expander("View Application & Prediction"):
                st.write("### 📌 Application Details")
                for key, val in input_data.items():
                    st.write(f"**{key}**: {val}")
                st.write("### 🎯 Prediction")
                for key, val in result.items():
                    if key == "error":
                        st.error(f"Prediction Error: {val}")
                    else:
                        st.write(f"**{key}**: {val}")

# -------------------------------
# ℹ️ Model Metadata
# -------------------------------
st.markdown("---")
st.header("ℹ️ Model Details")
st.write(f"**Model**: {predictor.model_name}")
st.write("**Version**: 1.0")
st.write("**Trained on**: 2025-05-30 18:48:00")
st.write("**Author**: Rovhona Mudau")
