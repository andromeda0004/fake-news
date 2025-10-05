# Retrain the fake news model with improved settings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("Loading data...")
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake['target'] = 'fake'
true['target'] = 'true'

# Combine
data = pd.concat([fake, true]).reset_index(drop=True)

print(f"Total articles: {len(data)}")
print(f"Fake articles: {len(fake)}")
print(f"True articles: {len(true)}")

# Shuffle
from sklearn.utils import shuffle
data = shuffle(data, random_state=42)
data = data.reset_index(drop=True)

# Drop date and title, keep only text
if 'date' in data.columns:
    data.drop(["date"], axis=1, inplace=True)
if 'title' in data.columns:
    data.drop(["title"], axis=1, inplace=True)
if 'subject' in data.columns:
    data.drop(["subject"], axis=1, inplace=True)

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop])
    
    return text

print("\nPreprocessing text...")
data['text'] = data['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], 
    data['target'], 
    test_size=0.2, 
    random_state=42,
    stratify=data['target']
)

print(f"\nTraining set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Try multiple models and pick the best

print("\n" + "="*80)
print("Training Logistic Regression Model...")
print("="*80)

pipe_lr = Pipeline([
    ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

pipe_lr.fit(X_train, y_train)
pred_lr = pipe_lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)

print(f"Logistic Regression Accuracy: {acc_lr*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred_lr))

print("\n" + "="*80)
print("Training Random Forest Model...")
print("="*80)

pipe_rf = Pipeline([
    ('vect', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

pipe_rf.fit(X_train, y_train)
pred_rf = pipe_rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)

print(f"Random Forest Accuracy: {acc_rf*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred_rf))

# Choose the best model
if acc_lr >= acc_rf:
    print("\n" + "="*80)
    print("Saving Logistic Regression Model (Best Performance)")
    print("="*80)
    joblib.dump(pipe_lr, "fake_news_model_improved.joblib")
    print(f"Model saved with {acc_lr*100:.2f}% accuracy")
    best_model = pipe_lr
else:
    print("\n" + "="*80)
    print("Saving Random Forest Model (Best Performance)")
    print("="*80)
    joblib.dump(pipe_rf, "fake_news_model_improved.joblib")
    print(f"Model saved with {acc_rf*100:.2f}% accuracy")
    best_model = pipe_rf

# Test with custom examples
print("\n" + "="*80)
print("Testing with Custom Examples")
print("="*80)

test_examples = [
    ("NASA's Perseverance rover has successfully collected its first samples of Martian rock, which scientists hope will provide clues about the planet's ancient environment.", "true"),
    ("Scientists confirm that drinking bleach cures COVID-19, according to a new study.", "fake"),
    ("The United Nations General Assembly adopted a resolution on climate change.", "true"),
    ("NASA admits the moon landing was filmed in a Hollywood studio.", "fake"),
]

for text, expected in test_examples:
    processed = preprocess_text(text)
    pred = best_model.predict([processed])[0]
    confidence = best_model.predict_proba([processed]).max() * 100
    status = "✓" if pred == expected else "✗"
    print(f"{status} Expected: {expected:5s} | Predicted: {pred:5s} | Confidence: {confidence:.1f}% | {text[:60]}...")

print("\n" + "="*80)
print("Done! Use 'fake_news_model_improved.joblib' in your app")
print("="*80)
