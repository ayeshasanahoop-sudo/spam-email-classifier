"""
Spam Email Classifier - Model Training Script
Uses TF-IDF + Naive Bayes (best combo for text classification)
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# ── Built-in stopwords (no NLTK download needed) ─────────────────────────────
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

# ── Synthetic training data ───────────────────────────────────────────────────
SPAM_SAMPLES = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "FREE MONEY!!! Act now and earn $5000 per week working from home!!!",
    "You have been selected for a special offer. Buy now and save 90%!",
    "URGENT: Your account has been compromised. Verify now at this link.",
    "Get rich quick! Invest in our guaranteed cryptocurrency scheme today.",
    "You won the lottery! Send us your bank details to claim your prize.",
    "Cheap medications online! No prescription needed. Order now!",
    "Make money fast! Our secret formula guarantees $10000 monthly!",
    "Dear winner, you have been selected. Provide your credit card info.",
    "Increase your size overnight! 100% natural pills with no side effects!",
    "Nigerian prince needs your help to transfer $50 million dollars urgently",
    "CLICK HERE for FREE iPhone! Limited offer expires in 24 hours!!!",
    "Earn $500 daily from home! No experience needed! Sign up FREE!!!",
    "Your PayPal account needs verification. Click this link immediately.",
    "Buy cheap Rolex watches, designer bags at 99% discount! Limited stock!",
    "Congratulations! You are our lucky winner. Claim your reward now!",
    "FINAL NOTICE: You owe taxes. Call immediately to avoid arrest warrant.",
    "Hot singles in your area are waiting to meet you! Click here now!",
    "Lose 30 pounds in 30 days with our miracle diet pill! No exercise needed!",
    "Exclusive investment opportunity! Guaranteed 300% returns in one month!",
    "Your computer is infected with virus! Download our FREE cleaner now!",
    "Work from home and earn thousands! No skills required! Start today!",
    "Free gift cards available! Survey takes 2 minutes! Claim yours now!",
    "You've been pre-approved for a $50,000 loan with no credit check!",
    "Discount Viagra available without prescription! Order online now!",
    "Winner! You've been selected for our exclusive loyalty program reward.",
    "Urgent account verification required. Failure to verify will result suspension.",
    "Make real money online with binary options! Guaranteed success formula!",
    "Your email was selected for free prize. Reply with your personal details.",
    "Refinance your mortgage today! Lowest rates guaranteed! Act now!",
    "SPECIAL OFFER: Buy one get ten free! Only for selected customers today!",
    "Dear customer your account will close unless you verify personal info.",
    "Earn commission by just sharing links! $100 per referral! Join free!",
    "Unclaimed package waiting. Provide payment for delivery to release it.",
    "Alert: Suspicious login detected. Verify your identity using link below.",
]

HAM_SAMPLES = [
    "Hi John, can we schedule a meeting tomorrow at 3pm to discuss the project?",
    "Please find attached the quarterly report for your review.",
    "Happy birthday! Hope you have a wonderful day with your family.",
    "The team meeting has been rescheduled to Friday at 2pm in conference room B.",
    "Thanks for your help with the presentation yesterday. It went really well!",
    "Reminder: dentist appointment on Wednesday at 10am.",
    "Can you please send me the updated version of the budget spreadsheet?",
    "I'll be out of office next week. Please contact Sarah for urgent matters.",
    "Great job on the project submission! The client was very impressed.",
    "Looking forward to seeing you at the conference next month in Mumbai.",
    "The new policy document has been uploaded to the shared drive.",
    "Could you review my pull request when you get a chance? No rush.",
    "Dinner tonight at 7pm? Let me know if you're available!",
    "Please submit your timesheet by end of day Friday.",
    "The library books you requested are now available for pickup.",
    "Your package has been shipped. Estimated delivery is Thursday.",
    "Thank you for attending our webinar. Here are the presentation slides.",
    "Interview confirmation: Monday 10am for Software Engineer position at TCS.",
    "The server maintenance will occur this Sunday from 2am to 4am IST.",
    "Payment received for invoice #1234. Thank you for your business.",
    "Your subscription renewal is due next month. No action required now.",
    "Team lunch tomorrow at 1pm. We're going to the new restaurant on MG Road.",
    "Please review the attached contract and let me know your thoughts.",
    "The project deadline has been extended by one week due to client request.",
    "Your flight booking confirmation: Hyderabad to Bangalore, Friday 6pm.",
    "Hi Mom, just wanted to check in. How are you feeling?",
    "The library has a new collection of programming books you might enjoy.",
    "Congrats on your promotion! Well deserved after all your hard work.",
    "Reminder: team standup at 9am. Please update your tasks in Jira beforehand.",
    "The code review has been completed. A few minor suggestions in the comments.",
    "Your annual performance review is scheduled for next Tuesday at 11am.",
    "Thanks for referring your colleague. We'll be in touch with them shortly.",
    "Please find the minutes from today's meeting for your reference.",
    "The canteen menu has been updated. New items available from Monday.",
    "Your application for leave from Dec 24 to Jan 2 has been approved.",
]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\b\d+\b', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

# ── Build dataset ─────────────────────────────────────────────────────────────
def build_dataset():
    texts = SPAM_SAMPLES + HAM_SAMPLES
    labels = [1] * len(SPAM_SAMPLES) + [0] * len(HAM_SAMPLES)
    df = pd.DataFrame({'text': texts, 'label': labels})
    df['text_clean'] = df['text'].apply(preprocess_text)
    return df

# ── Train & evaluate ──────────────────────────────────────────────────────────
def train_and_evaluate():
    print("=" * 55)
    print("  SPAM EMAIL CLASSIFIER - MODEL TRAINING")
    print("=" * 55)

    df = build_dataset()
    X = df['text_clean']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nDataset: {len(df)} samples | "
          f"Spam: {y.sum()} | Ham: {(y==0).sum()}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    models = {
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                      min_df=1, sublinear_tf=True)),
            ('clf', MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                      min_df=1, sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ]),
        "Linear SVM": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                      min_df=1, sublinear_tf=True)),
            ('clf', LinearSVC(C=1.0, max_iter=1000))
        ]),
    }

    results = {}
    best_model_name = None
    best_f1 = 0

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm
        }

        print(f"[{name}]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}\n")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    print(f"Best Model: {best_model_name} (F1={best_f1:.4f})")

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_pipeline, "model/spam_classifier.pkl")

    meta = {
        "best_model": best_model_name,
        "metrics": results[best_model_name],
        "all_models": results,
        "classes": {0: "ham", 1: "spam"},
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    with open("model/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nModel saved to model/spam_classifier.pkl")
    print("Metadata saved to model/model_meta.json")
    print("=" * 55)
    return best_pipeline, meta

if __name__ == "__main__":
    train_and_evaluate()
