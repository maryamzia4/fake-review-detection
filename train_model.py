# train_model.py

import numpy as np
import pandas as pd
import string
import nltk
import joblib
import warnings
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from utils import text_process

# Setup
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv('fake reviews dataset.csv')  # Your CSV file
df['text_'] = df['text_'].astype(str)
df.dropna(inplace=True)

# Check class distribution (print for debug)
print("Class distribution before preprocessing:")
print(df['label'].value_counts())

# Preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    try:
        tokens = wordpunct_tokenize(text)
        cleaned = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
        stemmed = [stemmer.stem(word) for word in cleaned]
        lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
        return ' '.join(lemmatized)
    except:
        return ''

df['text_'] = df['text_'].apply(preprocess)
df = df[df['text_'].str.strip() != '']

# Check class distribution after preprocessing
print("Class distribution after preprocessing:")
print(df['label'].value_counts())

# Save cleaned data (optional)
df.to_csv('preprocessed_fake_reviews.csv', index=False)

# Train/Test split
X = df['text_']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Simple analyzer function for CountVectorizer
def text_process(review):
    nopunc = ''.join([char for char in review if char not in string.punctuation])
    return [word for word in nopunc.split() if word.lower() not in stop_words]

# Build pipeline with SVM classifier and balanced class weight
model = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LinearSVC(class_weight='balanced', max_iter=10000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, 'fake_review_detector_svm.pkl')
print("✅ Model saved as fake_review_detector_svm.pkl")
