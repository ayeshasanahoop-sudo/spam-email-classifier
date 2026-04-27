# 📧 SpamShield — AI Spam Email Classifier

> Minor Project | B.E. Information Technology | NSAKCET Hyderabad

A full-stack machine learning web application that classifies emails as **SPAM** or **HAM (legitimate)** using Natural Language Processing and a Naive Bayes classifier.

---

## 🚀 Live Demo

> Open `frontend/index.html` directly in any browser — it works without the backend using a built-in heuristic engine.

To use the full ML backend:
```bash
pip install -r requirements.txt
python model/train_model.py
python backend/app.py
```
Then open `frontend/index.html`.

---

## 📂 Project Structure

```
spam-email-classifier/
├── backend/
│   └── app.py              ← Flask REST API (4 endpoints)
├── frontend/
│   └── index.html          ← Complete single-file UI
├── model/
│   ├── train_model.py      ← Training script
│   ├── spam_classifier.pkl ← Trained model (generated)
│   └── model_meta.json     ← Metrics & metadata (generated)
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### ML Pipeline
```
Raw Email Text
     ↓
Text Preprocessing
  • Lowercase
  • Remove URLs → token 'url'
  • Remove numbers → token 'num'
  • Remove punctuation
  • Remove stopwords
     ↓
TF-IDF Vectorization
  • Unigrams + Bigrams (n-gram range 1-2)
  • Max 5,000 features
  • Sublinear TF scaling
     ↓
Multinomial Naive Bayes
  • Alpha = 0.1 (Laplace smoothing)
     ↓
SPAM / HAM + Confidence Score
```

### Models Compared

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Naive Bayes** ✅ | 83.3% | 80.0% | 88.9% | **84.2%** |
| Logistic Regression | 83.3% | 80.0% | 88.9% | 84.2% |
| Linear SVM | 83.3% | 80.0% | 88.9% | 84.2% |

---

## 🔌 API Endpoints

### `POST /predict`
Classify a single email.

**Request:**
```json
{ "text": "Congratulations! You won a free iPhone! Click now!" }
```

**Response:**
```json
{
  "label": "spam",
  "is_spam": true,
  "confidence": 97.7,
  "spam_probability": 97.7,
  "ham_probability": 2.3,
  "risk_level": "HIGH",
  "spam_indicators": ["free", "win", "click", "congratulations"],
  "word_count": 9,
  "char_count": 52
}
```

### `POST /predict-batch`
Classify up to 50 emails at once.

### `GET /model-info`
Returns model metrics and metadata.

### `GET /health`
Health check endpoint.

---

## ✨ Features

- 🔍 **Single email analysis** with confidence score and risk level
- 📦 **Batch mode** — analyze up to 10 emails at once
- 📊 **Model info panel** — view accuracy, precision, recall, F1
- ⚡ **Works without backend** — built-in client-side demo mode
- 🎨 **Dark terminal-style UI** — professional and clean
- 📱 **Responsive** — works on mobile and desktop

---

## 🛠 Skills Demonstrated

| Category | Skills |
|---|---|
| **ML / NLP** | Text preprocessing, TF-IDF, Naive Bayes, Logistic Regression, SVM |
| **Python** | Scikit-learn, Pandas, NumPy, Joblib, Regex |
| **Backend** | Flask, REST API design, CORS, JSON |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, Fetch API |

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/ayeshasanahoop-sudo/spam-email-classifier.git
cd spam-email-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python model/train_model.py

# 4. Start the API server
python backend/app.py

# 5. Open frontend
# Open frontend/index.html in your browser
# OR serve with: python -m http.server 8080 (inside frontend/)
```

---

## 👩‍💻 Author

**Ayesha**  
B.E. Information Technology — 3rd Year  
Nawab Shah Alam Khan College of Engineering & Technology (NSAKCET), Hyderabad  

- GitHub: [@ayeshasanahoop-sudo](https://github.com/ayeshasanahoop-sudo)

---

*Minor Project submission — Spam Email Classifier using Machine Learning*
