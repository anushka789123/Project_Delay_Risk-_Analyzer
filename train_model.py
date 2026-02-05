import pandas as pd
import numpy as np
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


tasks = pd.read_csv("Dataset/tasks.csv")
messages = pd.read_csv("Dataset/messages.csv")

for col in ["start_date", "due_date", "completed_date"]:
    tasks[col] = pd.to_datetime(tasks[col])


tasks["delay_days"] = (tasks["completed_date"] - tasks["due_date"]).dt.days
tasks["delay_risk"] = (tasks["delay_days"] > 2).astype(int)


msg_agg = (
    messages.groupby("task_id")["message"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

data = tasks.merge(msg_agg, on="task_id")


data["task_duration"] = (data["due_date"] - data["start_date"]).dt.days

X_text = data["message"]
X_num = data[["workload_hours", "task_duration"]]
y = data["delay_risk"]


X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text,
    X_num,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=200
)

X_text_train_vec = tfidf.fit_transform(X_text_train)
X_text_test_vec = tfidf.transform(X_text_test)

X_train = np.hstack([X_num_train.values, X_text_train_vec.toarray()])
X_test = np.hstack([X_num_test.values, X_text_test_vec.toarray()])

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("🎯 Model Accuracy:", round(accuracy, 3))

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:50])

joblib.dump(shap_values, "shap_values.pkl")

print(" Model + SHAP saved successfully")
