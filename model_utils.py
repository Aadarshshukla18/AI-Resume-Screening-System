import joblib
import numpy as np

# Load models
xgb_model = joblib.load("xgb_resume_model.pkl")
sbert_model = joblib.load("sbert_model.pkl")


def predict_match_score(resume_text, job_description):

    resume_embedding = sbert_model.encode([resume_text])
    job_embedding = sbert_model.encode([job_description])

    features = np.hstack((resume_embedding, job_embedding))

    score = xgb_model.predict(features)

    return float(score[0])