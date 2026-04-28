"""
Spam Email Classifier - Streamlit App
Single file - Frontend + Backend combined!
"""

import re
import joblib
import numpy as np
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield — AI Email Classifier",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        font-family: 'Courier New', monospace;
    }
    .spam-color { color: #ff4466; }
    .ham-color  { color: #00e5a0; }

    .subtitle {
        text-align: center;
        color: #7a7a9a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Badge */
    .badge {
        text-align: center;
        margin-bottom: 1rem;
    }
    .badge span {
        background: #1a1a25;
        border: 1px solid #2a2a3a;
        border-radius: 999px;
        padding: 4px 16px;
        font-size: 12px;
        color: #7c6dfa;
        font-family: 'Courier New', monospace;
    }

    /* Result cards */
    .spam-card {
        background: rgba(255,68,102,0.08);
        border: 1px solid rgba(255,68,102,0.4);
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
    }
    .ham-card {
        background: rgba(0,229,160,0.07);
        border: 1px solid rgba(0,229,160,0.3);
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
    }

    /* Stat boxes */
    .stats-row {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }
    .stat-box {
        background: #0a0a0f;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        flex: 1;
    }
    .stat-val { font-size: 22px; font-weight: 700; font-family: 'Courier New', monospace; }
    .stat-key { font-size: 10px; color: #7a7a9a; letter-spacing: 0.1em; margin-top: 4px; }

    /* Indicator tags */
    .tag {
        display: inline-block;
        background: rgba(255,68,102,0.15);
        border: 1px solid rgba(255,68,102,0.3);
        border-radius: 6px;
        color: #ff4466;
        font-size: 12px;
        padding: 3px 10px;
        margin: 3px;
        font-family: 'Courier New', monospace;
    }

    /* Section divider */
    .section-label {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #7a7a9a;
        letter-spacing: 0.15em;
        margin-bottom: 8px;
    }

    /* Metric styling */
    .metric-row {
        display: flex; gap: 10px; margin: 10px 0;
    }
    .metric-card {
        background: #0a0a0f;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 14px 18px;
        flex: 1;
    }
    .metric-name { font-size: 11px; color: #7a7a9a; font-family: 'Courier New', monospace; }
    .metric-val  { font-size: 24px; font-weight: 700; color: #7c6dfa; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Stopwords ─────────────────────────────────────────────────────────────────
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

SPAM_KEYWORDS = [
    "free","win","winner","won","prize","gift","claim","offer","discount",
    "click","urgent","guaranteed","money","cash","earn","income","profit",
    "opportunity","limited","expires","act now","congratulations","selected",
    "verify","account","suspended","password","credit","loan","investment",
    "bitcoin","crypto","cheap","buy now","order","million","lottery","!!!",
    "pills","medication","weight loss","miracle","100%","nigerian","prince"
]

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
   model_path = Path(__file__).parent / "model" / "spam_classifier.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\b\d+\b', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

# ── Classify ──────────────────────────────────────────────────────────────────
def classify(text, model):
    cleaned = preprocess(text)
    pred = model.predict([cleaned])[0]
    try:
        proba = model.predict_proba([cleaned])[0]
        spam_prob = float(proba[1])
    except:
        decision = model.decision_function([cleaned])[0]
        spam_prob = float(1 / (1 + np.exp(-decision)))

    ham_prob = 1 - spam_prob
    confidence = max(spam_prob, ham_prob) * 100
    indicators = [k for k in SPAM_KEYWORDS if k in text.lower()][:6]

    risk = "🔴 HIGH" if spam_prob >= 0.85 else "🟡 MEDIUM" if spam_prob >= 0.55 else "🟢 LOW"

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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge"><span>🟢 SPAMSHIELD v1.0 — ML POWERED</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="main-title">
    Detect <span class="spam-color">Spam</span>.<br/>
    Protect <span class="ham-color">Inbox</span>.
</div>
<p class="subtitle">AI Email Classifier · TF-IDF + Naive Bayes · Built by Ayesha, NSAKCET Hyderabad</p>
""", unsafe_allow_html=True)

st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `python model/train_model.py` first.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Email", "📦 Batch Mode", "📊 Model Info"])

# ════════════════════════════════════════════════════════
# TAB 1 — Single Email
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Paste your email text below:")

    # Quick example buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🚨 Spam Ex 1"):
            st.session_state['email_text'] = "Congratulations! You've WON a FREE iPhone and $5000 cash! Click here NOW to CLAIM your prize before it EXPIRES!!!"
    with col2:
        if st.button("🚨 Spam Ex 2"):
            st.session_state['email_text'] = "URGENT: Your bank account has been suspended! Verify your identity immediately by clicking this link or your account will be permanently closed."
    with col3:
        if st.button("✅ Ham Ex 1"):
            st.session_state['email_text'] = "Hi, can we reschedule tomorrow's meeting to 3pm? Let me know if that works for you. Thanks!"
    with col4:
        if st.button("✅ Ham Ex 2"):
            st.session_state['email_text'] = "Please find attached the project report for your review. Let me know if you have any feedback."

    email_text = st.text_area(
        label="Email Text",
        value=st.session_state.get('email_text', ''),
        height=180,
        placeholder="Paste email subject + body here...",
        label_visibility="collapsed"
    )

    st.markdown(f"<p style='color:#7a7a9a;font-size:12px;'>{len(email_text)} characters · {len(email_text.split())} words</p>", unsafe_allow_html=True)

    analyze_btn = st.button("🔍 ANALYZE EMAIL", type="primary", use_container_width=True)

    if analyze_btn:
        if not email_text.strip():
            st.warning("Please paste an email text first!")
        else:
            with st.spinner("Analyzing..."):
                result = classify(email_text, model)

            # Result card
            if result['is_spam']:
                st.markdown(f"""
                <div class="spam-card">
                    <h2 style="color:#ff4466;margin:0;">🚨 SPAM DETECTED</h2>
                    <p style="color:#7a7a9a;margin:4px 0 16px;">Risk Level: {result['risk']} · This email shows spam characteristics</p>

                    <div class="section-label">// SPAM PROBABILITY</div>
                    <div style="background:#0a0a0f;border-radius:999px;height:10px;margin-bottom:16px;">
                        <div style="width:{result['spam_prob']}%;background:linear-gradient(90deg,#ff4466,#ff7090);height:10px;border-radius:999px;"></div>
                    </div>

                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-val" style="color:#ff4466">{result['confidence']}%</div>
                            <div class="stat-key">CONFIDENCE</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" style="color:#ff4466">{result['spam_prob']}%</div>
                            <div class="stat-key">SPAM PROB</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val">{result['words']}</div>
                            <div class="stat-key">WORDS</div>
                        </div>
                    </div>

                    {'<div class="section-label">// SPAM INDICATORS FOUND</div>' + ''.join([f'<span class="tag">{i}</span>' for i in result['indicators']]) if result['indicators'] else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ham-card">
                    <h2 style="color:#00e5a0;margin:0;">✅ LEGITIMATE EMAIL</h2>
                    <p style="color:#7a7a9a;margin:4px 0 16px;">Risk Level: {result['risk']} · This email appears safe</p>

                    <div class="section-label">// HAM PROBABILITY</div>
                    <div style="background:#0a0a0f;border-radius:999px;height:10px;margin-bottom:16px;">
                        <div style="width:{result['ham_prob']}%;background:linear-gradient(90deg,#00e5a0,#40ffc0);height:10px;border-radius:999px;"></div>
                    </div>

                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-val" style="color:#00e5a0">{result['confidence']}%</div>
                            <div class="stat-key">CONFIDENCE</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val" style="color:#00e5a0">{result['ham_prob']}%</div>
                            <div class="stat-key">HAM PROB</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val">{result['words']}</div>
                            <div class="stat-key">WORDS</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Batch Mode
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Analyze multiple emails at once (one per line):")
    batch_text = st.text_area(
        "Batch emails",
        height=220,
        placeholder="Email 1 text here...\n---\nEmail 2 text here...\n---\nEmail 3 text here...",
        label_visibility="collapsed"
    )

    if st.button("📦 ANALYZE ALL", type="primary", use_container_width=True):
        if not batch_text.strip():
            st.warning("Please add some emails first!")
        else:
            emails = [e.strip() for e in batch_text.split("---") if e.strip()]
            if not emails:
                emails = [e.strip() for e in batch_text.split("\n") if e.strip()]

            results = [classify(e, model) for e in emails]
            spam_count = sum(1 for r in results if r['is_spam'])
            ham_count = len(results) - spam_count

            # Summary
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Emails", len(results))
            c2.metric("🚨 Spam", spam_count)
            c3.metric("✅ Legitimate", ham_count)

            st.divider()

            for i, (email, result) in enumerate(zip(emails, results)):
                icon  = "🚨" if result['is_spam'] else "✅"
                color = "#ff4466" if result['is_spam'] else "#00e5a0"
                label = "SPAM" if result['is_spam'] else "HAM"
                preview = email[:80] + "..." if len(email) > 80 else email
                st.markdown(f"""
                <div style="background:#12121a;border:1px solid #2a2a3a;border-radius:10px;
                            padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px;">
                    <span style="color:{color};font-family:monospace;font-weight:700;
                                 background:rgba(0,0,0,0.3);padding:2px 8px;border-radius:5px;">
                        {icon} {label}
                    </span>
                    <span style="color:#7a7a9a;font-size:13px;">{result['confidence']}% confident</span>
                    <span style="color:#e8e8f0;font-size:13px;flex:1;">{preview}</span>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Model Info
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### ML Pipeline")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:12px 0;">
        <div style="background:#1a1a25;border:1px solid #2a2a3a;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:12px;">📥 Raw Email</div>
        <span style="color:#7a7a9a;">→</span>
        <div style="background:#1a1a25;border:1px solid #2a2a3a;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:12px;">🧹 Preprocessing</div>
        <span style="color:#7a7a9a;">→</span>
        <div style="background:#1a1a25;border:1px solid #2a2a3a;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:12px;">📊 TF-IDF</div>
        <span style="color:#7a7a9a;">→</span>
        <div style="background:#1a1a25;border:1px solid #2a2a3a;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:12px;">🤖 Naive Bayes</div>
        <span style="color:#7a7a9a;">→</span>
        <div style="background:#1a1a25;border:1px solid #2a2a3a;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:12px;">✅ Result</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "83.3%")
    col2.metric("Precision", "80.0%")
    col3.metric("Recall",    "88.9%")
    col4.metric("F1 Score",  "84.2%")

    st.divider()
    st.markdown("#### Models Compared")

    import pandas as pd
    df = pd.DataFrame({
        "Model": ["✅ Naive Bayes (Selected)", "Logistic Regression", "Linear SVM"],
        "Accuracy": ["83.3%", "83.3%", "83.3%"],
        "Precision": ["80.0%", "80.0%", "80.0%"],
        "Recall": ["88.9%", "88.9%", "88.9%"],
        "F1 Score": ["84.2%", "84.2%", "84.2%"]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Skills Demonstrated")
    skills = ["Text Preprocessing", "TF-IDF Feature Extraction",
              "Naive Bayes Classifier", "Logistic Regression",
              "Linear SVM", "Scikit-learn Pipeline",
              "Model Comparison", "Streamlit Deployment"]
    cols = st.columns(4)
    for i, skill in enumerate(skills):
        cols[i % 4].success(skill)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center;color:#7a7a9a;font-size:12px;font-family:monospace;">
    Built by <strong style="color:#e8e8f0;">Ayesha</strong> ·
    B.E. IT — NSAKCET, Hyderabad ·
    Spam Email Classifier Minor Project
</p>
""", unsafe_allow_html=True)
