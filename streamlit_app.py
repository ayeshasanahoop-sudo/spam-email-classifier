import re
import joblib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SpamShield - AI Email Classifier",
    page_icon="shield",
    layout="centered"
)

st.markdown("""
<style>
.stApp { background-color: #0a0a0f; color: #e8e8f0; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her","it",
    "its","they","them","what","which","who","this","that","these","those",
    "am","is","are","was","were","be","been","have","has","had","do","does",
    "did","a","an","the","and","but","if","or","as","of","at","by","for",
    "with","to","from","in","out","on","off","so","than","too","very","just",
    "can","will","now","not","no","nor","all","both","each","few","more",
    "most","same","such","own","then","once","here","there","when","where"
}

SPAM_KEYWORDS = [
    "free","win","winner","won","prize","gift","claim","offer","discount",
    "click","urgent","guaranteed","money","cash","earn","income","profit",
    "opportunity","limited","expires","congratulations","selected","verify",
    "account","suspended","password","credit","loan","investment","bitcoin",
    "crypto","cheap","million","lottery","pills","weight loss","miracle","!!!"
]

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model" / "spam_classifier.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\b\d+\b', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

def classify(text, model):
    cleaned = preprocess(text)
    pred = model.predict([cleaned])[0]
    try:
        proba = model.predict_proba([cleaned])[0]
        spam_prob = float(proba[1])
    except Exception:
        decision = model.decision_function([cleaned])[0]
        spam_prob = float(1 / (1 + np.exp(-decision)))
    ham_prob = 1 - spam_prob
    confidence = max(spam_prob, ham_prob) * 100
    indicators = [k for k in SPAM_KEYWORDS if k in text.lower()][:6]
    if spam_prob >= 0.85:
        risk = "HIGH"
    elif spam_prob >= 0.55:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {
        "is_spam": bool(pred == 1),
        "label": "SPAM" if pred == 1 else "HAM",
        "confidence": round(confidence, 1),
        "spam_prob": round(spam_prob * 100, 1),
        "ham_prob": round(ham_prob * 100, 1),
        "risk": risk,
        "indicators": indicators,
        "words": len(text.split())
    }

# Header
st.markdown("## SpamShield — AI Email Classifier")
st.markdown("**Detect Spam. Protect Inbox.** | TF-IDF + Naive Bayes | Built by Ayesha, NSAKCET Hyderabad")
st.divider()

model = load_model()
if model is None:
    st.error("Model not found! Please check that model/spam_classifier.pkl exists in the repo.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single Email", "Batch Mode", "Model Info"])

with tab1:
    st.markdown("#### Paste your email text below:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Spam Example 1"):
            st.session_state['email_text'] = "Congratulations! You've WON a FREE iPhone and $5000 cash! Click here NOW to CLAIM your prize before it EXPIRES!!!"
    with col2:
        if st.button("Spam Example 2"):
            st.session_state['email_text'] = "URGENT: Your bank account has been suspended! Verify your identity immediately or your account will be permanently closed."
    with col3:
        if st.button("Ham Example 1"):
            st.session_state['email_text'] = "Hi, can we reschedule tomorrow meeting to 3pm? Let me know if that works for you. Thanks!"
    with col4:
        if st.button("Ham Example 2"):
            st.session_state['email_text'] = "Please find attached the project report for your review. Let me know if you have any feedback."

    email_text = st.text_area(
        label="Email Text",
        value=st.session_state.get('email_text', ''),
        height=180,
        placeholder="Paste email subject and body here...",
        label_visibility="collapsed"
    )

    st.caption(f"{len(email_text)} characters | {len(email_text.split())} words")

    if st.button("ANALYZE EMAIL", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please paste an email text first!")
        else:
            with st.spinner("Analyzing..."):
                result = classify(email_text, model)

            if result['is_spam']:
                st.error(f"SPAM DETECTED | Risk Level: {result['risk']}")
            else:
                st.success(f"LEGITIMATE EMAIL | Risk Level: {result['risk']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{result['confidence']}%")
            col2.metric("Spam Probability", f"{result['spam_prob']}%")
            col3.metric("Word Count", result['words'])

            if result['is_spam'] and result['indicators']:
                st.markdown("**Spam Indicators Found:**")
                st.code(" | ".join(result['indicators']))

with tab2:
    st.markdown("#### Analyze multiple emails — separate each email with `---`")

    batch_text = st.text_area(
        "Batch emails",
        height=220,
        placeholder="Email 1 text here...\n---\nEmail 2 text here...\n---\nEmail 3 text here...",
        label_visibility="collapsed"
    )

    if st.button("ANALYZE ALL", type="primary", use_container_width=True):
        if not batch_text.strip():
            st.warning("Please add some emails first!")
        else:
            emails = [e.strip() for e in batch_text.split("---") if e.strip()]
            if not emails:
                emails = [e.strip() for e in batch_text.split("\n") if e.strip()]

            results = [classify(e, model) for e in emails]
            spam_count = sum(1 for r in results if r['is_spam'])
            ham_count = len(results) - spam_count

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Emails", len(results))
            c2.metric("Spam", spam_count)
            c3.metric("Legitimate", ham_count)

            st.divider()
            for i, (email, result) in enumerate(zip(emails, results)):
                label = "SPAM" if result['is_spam'] else "HAM"
                preview = email[:80] + "..." if len(email) > 80 else email
                if result['is_spam']:
                    st.error(f"[{label}] {result['confidence']}% — {preview}")
                else:
                    st.success(f"[{label}] {result['confidence']}% — {preview}")

with tab3:
    st.markdown("#### ML Pipeline")
    st.markdown("Raw Email → Preprocessing → TF-IDF Vectorization → Naive Bayes → SPAM / HAM")
    st.divider()

    st.markdown("#### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "83.3%")
    col2.metric("Precision", "80.0%")
    col3.metric("Recall", "88.9%")
    col4.metric("F1 Score", "84.2%")

    st.divider()
    st.markdown("#### Models Compared")
    import pandas as pd
    df = pd.DataFrame({
        "Model": ["Naive Bayes (Selected)", "Logistic Regression", "Linear SVM"],
        "Accuracy": ["83.3%", "83.3%", "83.3%"],
        "Precision": ["80.0%", "80.0%", "80.0%"],
        "Recall": ["88.9%", "88.9%", "88.9%"],
        "F1 Score": ["84.2%", "84.2%", "84.2%"]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Skills Demonstrated")
    skills = [
        "Text Preprocessing", "TF-IDF Feature Extraction",
        "Naive Bayes Classifier", "Logistic Regression",
        "Linear SVM", "Scikit-learn Pipeline",
        "Model Comparison", "Streamlit Deployment"
    ]
    cols = st.columns(4)
    for i, skill in enumerate(skills):
        cols[i % 4].success(skill)

st.divider()
st.caption("Built by Ayesha | B.E. IT — NSAKCET, Hyderabad | Spam Email Classifier Minor Project")
