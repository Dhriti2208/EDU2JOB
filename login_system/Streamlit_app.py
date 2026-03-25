import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Edu2Job - Career Recommendation",
    page_icon="🎯",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("gb_model.pkl")
degree_encoder = joblib.load("degree_encoder.pkl")
spec_encoder = joblib.load("spec_encoder.pkl")
job_encoder = joblib.load("job_encoder.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background-color: #0e0e0e;
}

.main {
    background-color: #0e0e0e;
}

.block-container {
    padding-top: 3rem;
}

.card {
    background-color: #1c1c1c;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.6);
}

.title {
    text-align: center;
    color: #facc15;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 40px;
}

.stButton>button {
    background-color: #facc15;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #eab308;
    color: black;
}

.result-box {
    margin-top: 25px;
    padding: 25px;
    border: 2px solid #facc15;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    color: #facc15;
    font-weight: bold;
}

label, .stSelectbox label, .stNumberInput label {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">Edu2Job</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Transform your academic journey into a successful career path.</div>',
    unsafe_allow_html=True
)

# ---------------- MAIN CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

degree = st.selectbox("Degree", degree_encoder.classes_)
specialization = st.selectbox("Specialization", spec_encoder.classes_)
cgpa = st.number_input(
    "CGPA",
    min_value=5.00,
    max_value=10.00,
    step=0.01,
    format="%.2f"
)

predict_button = st.button("Get Career Recommendation")

if predict_button:
    degree_encoded = degree_encoder.transform([degree])[0]
    spec_encoded = spec_encoder.transform([specialization])[0]

    input_df = pd.DataFrame(
        [[degree_encoded, spec_encoded, cgpa]],
        columns=["Degree", "Specialization", "CGPA"]
    )

    prediction = model.predict(input_df)
    job_role = job_encoder.inverse_transform(prediction)[0]

    st.markdown(
        f'<div class="result-box">Recommended Career Path<br><br>{job_role}</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)