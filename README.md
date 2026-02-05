#  Project Delay Risk Analyzer

An end-to-end Machine Learning application that predicts **project task delay risk** using **task workload, duration, and team communication data**, with explainable insights using **SHAP** and an interactive **Streamlit dashboard**.

---

##  Project Overview

Project delays often occur due to a combination of workload imbalance, unrealistic timelines, and communication bottlenecks.  
This project analyzes historical task data and team messages to **predict whether a task is at high risk of delay** and **explain why** the model made that prediction.

The system combines **numerical features** (workload, duration) with **textual features** (team messages) using NLP.

---

## Key Features

- Predicts **High / Low Delay Risk**
- Uses **TF-IDF** to extract insights from team messages
- Machine Learning model trained using **Scikit-learn**
- **SHAP Explainability** to interpret predictions
- Interactive **Streamlit Web App**
  

---

## Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **NLP:** TF-IDF Vectorization  
- **Explainability:** SHAP  
- **Web App:** Streamlit  
- **Model Storage:** Joblib  

---

## Project Structure
Project Delay Risk Analyzer/

-Dataset/
    - tasks.csv # Task-level information
    - messages.csv # Team communication data

-train_model.py # Model training + SHAP generation
-app.py # Streamlit application
-model.pkl # Trained ML model
-tfidf.pkl # TF-IDF vectorizer
-shap_values.pkl # SHAP explanation values
-requirements.txt # Project dependencies


