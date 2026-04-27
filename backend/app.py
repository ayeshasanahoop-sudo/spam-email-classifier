"""
Spam Email Classifier - Flask Backend API
Endpoints:
  POST /predict       → classify single email
  POST /predict-batch → classify multiple emails
  GET  /model-info    → model metadata & metrics
  GET  /health        → health check
"""

import os
import re
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_classifier.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "model_meta.json")

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    model_meta = json.load(f)

print(f"[OK] Model loaded: {model_meta['best_model']}")
print(f"[OK] Accuracy: {model_meta['metrics']['accuracy']}")

# ── Stopwords (same as training) ──────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don","should",
    "now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn",
    "hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn"
}

# ── Spam indicator keywords (for explanation) ─────────────────────────────────
SPAM_KEYWORDS = [
    "free","win","winner","won","prize","gift","claim","offer","discount",
    "click","urgent","guaranteed","money","cash","earn","income","profit",
    "opportunity","limited","expires","act now","congratulations","selected",
    "verify","account","suspended","password","credit","loan","investment",
    "bitcoin","crypto","cheap","buy now","order","subscribe","unsubscribe",
    "million","billion","nigerian","prince","lottery","jackpot","casino",
    "pills","medication","viagra","weight loss","miracle","100%","!!!"
]

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\b\d+\b', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

def get_spam_indicators(text: str) -> list:
    text_lower = text.lower()
    found = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
    return found[:8]  # Return top 8

def classify_email(text: str) -> dict:
    if not text or not text.strip():
        return {"error": "Email text cannot be empty."}

    cleaned = preprocess_text(text)
    prediction = model.predict([cleaned])[0]

    # Get probability if model supports it
    try:
        proba = model.predict_proba([cleaned])[0]
        confidence = float(max(proba))
        spam_prob  = float(proba[1])
        ham_prob   = float(proba[0])
    except AttributeError:
        # LinearSVC doesn't have predict_proba
        decision = model.decision_function([cleaned])[0]
        spam_prob  = float(1 / (1 + np.exp(-decision)))
        ham_prob   = 1 - spam_prob
        confidence = max(spam_prob, ham_prob)

    label = "spam" if prediction == 1 else "ham"
    indicators = get_spam_indicators(text) if label == "spam" else []

    # Risk level
    if spam_prob >= 0.85:
        risk = "HIGH"
    elif spam_prob >= 0.55:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "label": label,
        "is_spam": bool(prediction == 1),
        "confidence": round(confidence * 100, 2),
        "spam_probability": round(spam_prob * 100, 2),
        "ham_probability": round(ham_prob * 100, 2),
        "risk_level": risk,
        "spam_indicators": indicators,
        "word_count": len(text.split()),
        "char_count": len(text)
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": model_meta["best_model"],
        "version": "1.0.0"
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "best_model": model_meta["best_model"],
        "metrics": model_meta["metrics"],
        "all_models": model_meta["all_models"],
        "training_samples": model_meta["training_samples"],
        "test_samples": model_meta["test_samples"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Request body must have a 'text' field."}), 400

    result = classify_email(data["text"])
    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "emails" not in data or not isinstance(data["emails"], list):
        return jsonify({"error": "Request body must have an 'emails' list."}), 400

    if len(data["emails"]) > 50:
        return jsonify({"error": "Max 50 emails per batch."}), 400

    results = []
    for i, email_text in enumerate(data["emails"]):
        res = classify_email(str(email_text))
        res["index"] = i
        results.append(res)

    spam_count = sum(1 for r in results if r.get("is_spam"))
    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "spam": spam_count,
            "ham": len(results) - spam_count,
            "spam_percentage": round(spam_count / len(results) * 100, 1)
        }
    })

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Spam Classifier API running on http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
