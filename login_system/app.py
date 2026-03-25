from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

users = {}
profiles = {}

# ------------------ LOAD ML MODEL ------------------

model = joblib.load("gb_model.pkl")
degree_encoder = joblib.load("degree_encoder.pkl")
spec_encoder = joblib.load("spec_encoder.pkl")
job_encoder = joblib.load("job_encoder.pkl")

# ------------------ ML-BASED JOB PREDICTION ------------------

def predict_top3_jobs(degree, specialization, cgpa):

    try:
        cgpa = float(cgpa)
    except:
        return []

    try:
        degree = degree.strip()
        specialization = specialization.strip().lower()

        # -------- Degree Normalization --------
        degree = next(
            (d for d in degree_encoder.classes_
             if d.lower() == degree.lower()),
            None
        )

        # -------- Specialization Normalization --------
        synonym_map = {
            "aiml": "Artificial Intelligence",
            "ai": "Artificial Intelligence",
            "ml": "Artificial Intelligence",
            "cs": "Computer Science",
            "cse": "Computer Science",
            "it": "Information Technology",
            "data": "Data Science"
        }

        # Map known short forms
        if specialization in synonym_map:
            specialization = synonym_map[specialization].lower()

        specialization = next(
            (s for s in spec_encoder.classes_
             if s.lower() == specialization),
            None
        )

        if not degree or not specialization:
            return []

        degree_encoded = degree_encoder.transform([degree])[0]
        spec_encoded = spec_encoder.transform([specialization])[0]

        input_df = pd.DataFrame(
            [[degree_encoded, spec_encoded, cgpa]],
            columns=["Degree", "Specialization", "CGPA"]
        )

        probabilities = model.predict_proba(input_df)[0]
        top_indices = probabilities.argsort()[::-1][:3]

        top_jobs = []

        for idx in top_indices:
            job_name = job_encoder.inverse_transform([idx])[0]
            confidence = round(probabilities[idx] * 100, 2)

            top_jobs.append({
                "job": job_name,
                "confidence": confidence
            })

        return top_jobs

    except:
        return []
# ------------------ PROFILE SETUP ------------------

def ensure_profile_exists(username):
    if username not in profiles:
        profiles[username] = {
            "name": "",
            "degree": "",
            "specialization": "",
            "cgpa": "",
            "skills": [],
            "photo": ""
        }

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            error = "Fields cannot be empty"
        elif username in users:
            error = "User already exists"
        else:
            users[username] = password
            ensure_profile_exists(username)
            flash("Registered successfully. Please login.")
            return redirect(url_for("login"))

    return render_template("register.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username in users and users[username] == password:
            session["user"] = username
            ensure_profile_exists(username)
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid credentials"

    return render_template("login.html", error=error)

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]
    ensure_profile_exists(user)

    if request.method == "POST":
        profiles[user]["name"] = request.form.get("name")
        profiles[user]["degree"] = request.form.get("degree")
        profiles[user]["specialization"] = request.form.get("specialization")
        profiles[user]["cgpa"] = request.form.get("cgpa")
        profiles[user]["skills"] = request.form.getlist("skills")

        photo = request.files.get("photo")
        if photo and photo.filename != "":
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            profiles[user]["photo"] = filename

        flash("Profile updated successfully")
        return redirect(url_for("dashboard"))

    return render_template("profile.html", profile=profiles[user])

@app.route("/view-profile")
def view_profile():
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]
    ensure_profile_exists(user)
    return render_template("view_profile.html", profile=profiles[user])

@app.route("/job-recommendation")
def job_recommendation():
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]
    ensure_profile_exists(user)

    profile = profiles[user]

    top_jobs = predict_top3_jobs(
        profile.get("degree"),
        profile.get("specialization"),
        profile.get("cgpa")
    )

    return render_template(
        "job_recommendation.html",
        profile=profile,
        top_jobs=top_jobs
    )

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)