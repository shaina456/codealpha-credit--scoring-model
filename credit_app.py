import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("credit_scoring_model.pkl")

model = load_model()

st.title("Credit Scoring Model")
st.write("Machine Learning Based Loan Risk Prediction")
st.divider()

st.subheader("Applicant Information")

with st.form("credit_form"):

    age = st.number_input("Age", 18, 100, 25)
    income = st.number_input("Annual Income", min_value=0, value=300000)
    emp_exp = st.number_input("Employment Experience (years)", 0, 80, 3)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
    interest_rate = st.number_input("Loan Interest Rate (%)", 0.0, 100.0, 10.0)
    credit_history = st.number_input("Credit History Length (years)", 0, 60, 5)
    credit_score = st.number_input("Credit Score", 300, 850, 650)

    gender = st.selectbox("Gender", ["female", "male"])

    education = st.selectbox(
        "Education",
        ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
    )

    home = st.selectbox(
        "Home Ownership",
        ["MORTGAGE", "OTHER", "OWN", "RENT"]
    )

    intent = st.selectbox(
        "Loan Purpose",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ]
    )

    previous_default = st.selectbox(
        "Previous Loan Defaults",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button("Predict Credit Risk")

if submitted:

    if income <= 0:
        st.error("Annual income must be greater than zero.")
    elif loan_amount <= 0:
        st.error("Loan amount must be greater than zero.")
    else:
        loan_percent_income = loan_amount / income

        # Start with all 27 trained input features
        input_data = pd.DataFrame(
            0.0,
            index=[0],
            columns=model.feature_names_in_
        )

        numeric_values = {
            "person_age": age,
            "person_income": income,
            "person_emp_exp": emp_exp,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": credit_history,
            "credit_score": credit_score
        }

        for column, value in numeric_values.items():
            if column in input_data.columns:
                input_data.loc[0, column] = value

        categorical_columns = [
            f"person_gender_{gender}",
            f"person_education_{education}",
            f"person_home_ownership_{home}",
            f"loan_intent_{intent}",
            f"previous_loan_defaults_on_file_{previous_default}"
        ]

        for column in categorical_columns:
            if column in input_data.columns:
                input_data.loc[0, column] = 1

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        class_1_index=list(model.classes_).index(1)
        risk_probability=probabilities[class_1_index]*100
        st.metric("Class 1 probability",f"{risk_probability:.2f}%")

        st.write("model classes", model.classes_)
        st.write("ALL probabilities",probabilities)
        st.write("predicted class",prediction)
    

        class_1_index = list(model.classes_).index(1)
        
        st.subheader("Prediction Result")
        

        if prediction == 1:
            st.warning("The model predicts Class 1.")
        else:
            st.success("The model predicts Class 0.")

        st.caption(
            "Class meanings must be verified against the dataset's "
            "loan_status definition. This is an educational prediction, "
            "not a real lending decision."
        )

        


