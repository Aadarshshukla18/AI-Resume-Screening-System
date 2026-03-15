import pandas as pd
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("resume_data.csv")

# Combine resume text
df["resume_text"] = (
    df["career_objective"].fillna("") + " " +
    df["skills"].fillna("") + " " +
    df["responsibilities"].fillna("")
)

# Job description text
df["job_text"] = df["skills_required"].fillna("")

# Target variable
y = df["matched_score"]

print("Loading Sentence-BERT model...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

resume_embeddings = sbert.encode(df["resume_text"].tolist())
job_embeddings = sbert.encode(df["job_text"].tolist())

# Combine features
X = np.hstack((resume_embeddings, job_embeddings))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)

print("Model MSE:", mse)

# Save model
joblib.dump(model, "xgb_resume_model.pkl")

print("Model saved successfully")