# AI Resume Screening System using NLP & Machine Learning

An intelligent **Resume Screening System** built using **Sentence-BERT, XGBoost, and Streamlit** that automatically analyzes resumes and predicts how well they match a given job description.

This project simulates a **real-world Applicant Tracking System (ATS)** used by companies to filter candidates efficiently.

---

## 🚀 Features

* 📄 Upload Resume in **PDF format**
* 🧠 Extract resume text automatically
* 🔎 Convert resume and job description into **Sentence-BERT embeddings**
* 🤖 Predict **resume–job match score using XGBoost**
* 📊 Display **match percentage and evaluation**
* 🌐 Interactive **Streamlit web application**

---

## 🧠 Machine Learning Pipeline

```
Resume Upload
      ↓
PDF Text Extraction
      ↓
Sentence-BERT Embeddings
      ↓
Feature Combination
      ↓
XGBoost Regression Model
      ↓
Match Score Prediction
      ↓
Streamlit Web Interface
```

---

## 🛠 Tech Stack

| Technology     | Usage                  |
| -------------- | ---------------------- |
| Python         | Core programming       |
| Sentence-BERT  | NLP embeddings         |
| XGBoost        | Machine learning model |
| Streamlit      | Web interface          |
| PyPDF2         | Resume text extraction |
| Scikit-learn   | ML utilities           |
| Pandas & NumPy | Data processing        |

---

## 📂 Project Structure

```
Resume-Analyzer-NLP
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── resume_data.csv        # Dataset
├── xgb_resume_model.pkl   # Trained ML model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone git https://github.com/Aadarshshukla18/AI-Resume-Screening-System.git
cd resume-screening-nlp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model

Run the training script to generate the ML model:

```bash
python train_model.py
```

This will create:

```
xgb_resume_model.pkl
```

---

## ▶️ Run the Web Application

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Example Output

```
Resume Match Score: 82%

✔ Strong Match
```

---

## 💡 Future Improvements

* Skill extraction using NLP
* Missing skill detection
* Resume ranking system
* Multi-resume comparison
* ATS-style dashboard
* Deployment on cloud (Streamlit Cloud / AWS)

---

## 🎯 Use Cases

* Resume screening automation
* Recruitment AI tools
* HR analytics systems
* NLP & ML portfolio projects

---

## 👨‍💻 Author

**Aadarsh Shukla** (**Data Scientist**)

Aspiring **AI/ML Engineer** 

---

⭐ If you like this project, consider giving it a **star on GitHub!**
