import streamlit as st
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Project Delay Risk Analyzer",
    layout="wide"
)


model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")


st.title("🚦 Project Delay Risk Analyzer")

st.write(
    "Predict project delay risk using **workload**, **task duration**, "
    "and **team communication messages**."
)


st.sidebar.header("📥 Task Inputs")

workload = st.sidebar.slider(
    "Workload Hours",
    min_value=5,
    max_value=80,
    value=40
)

duration = st.sidebar.slider(
    "Task Duration (Days)",
    min_value=1,
    max_value=30,
    value=10
)

message = st.sidebar.text_area(
    "Team Messages",
    value="Blocked due to dependency and waiting for approval"
)


if st.sidebar.button("Analyze Delay Risk"):

    
    text_vec = tfidf.transform([message]).toarray()

    
    X_input = np.hstack([[workload, duration], text_vec[0]]).reshape(1, -1)

    
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("🔍 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ High Delay Risk\n\nRisk Probability: {prob:.2f}")
    else:
        st.success(f"✅ Low Delay Risk\n\nConfidence: {1 - prob:.2f}")


    st.subheader("📊 Why this prediction? (SHAP Explanation)")

    explainer = shap.Explainer(model, X_input)
    shap_values = explainer(X_input)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.markdown(
        """
        **How to interpret this explanation:**
        - 🔴 Red features increase delay risk  
        - 🔵 Blue features reduce delay risk  
        - Longer bars indicate stronger influence
        """
    )


st.markdown("---")
st.caption(
    "Built using Python, NLP (TF-IDF), Machine Learning, SHAP, and Streamlit"
)
