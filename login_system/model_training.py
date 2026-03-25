import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("job_dataset.csv")

# ---------------- ENCODING ----------------
degree_encoder = LabelEncoder()
spec_encoder = LabelEncoder()
job_encoder = LabelEncoder()

df["Degree"] = degree_encoder.fit_transform(df["Degree"])
df["Specialization"] = spec_encoder.fit_transform(df["Specialization"])
df["JobRole"] = job_encoder.fit_transform(df["JobRole"])

# ---------------- FEATURES & TARGET ----------------
X = df[["Degree", "Specialization", "CGPA"]]
y = df["JobRole"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully.")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ---------------- SAVE MODEL & ENCODERS ----------------
joblib.dump(model, "gb_model.pkl")
joblib.dump(degree_encoder, "degree_encoder.pkl")
joblib.dump(spec_encoder, "spec_encoder.pkl")
joblib.dump(job_encoder, "job_encoder.pkl")

print("Model and encoders saved successfully.")