import streamlit as st
import joblib
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer

# Page title
st.set_page_config(page_title="AI Resume Screening System")

st.title("AI Resume Screening System")
st.write("Upload a resume and compare it with a job description.")

# Load models
xgb_model = joblib.load("xgb_resume_model.pkl")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to extract text from PDF
def extract_text_from_pdf(file):

    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


# Resume Upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job Description
job_description = st.text_area("Enter Job Description")


# Run prediction
if st.button("Check Match Score"):

    if uploaded_file is not None and job_description != "":

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_file)

        # Generate embeddings
        resume_embedding = sbert_model.encode([resume_text])
        job_embedding = sbert_model.encode([job_description])

        # Combine features
        features = np.hstack((resume_embedding, job_embedding))

        # Predict score
        score = xgb_model.predict(features)[0]

        # Display result
        st.subheader("Resume Match Score")

        st.write(f"{round(score*100,2)} %")

        if score > 0.75:
            st.success("Strong Match")

        elif score > 0.5:
            st.warning("Moderate Match")

        else:
            st.error("Low Match")

    else:

        st.warning("Please upload resume and enter job description")