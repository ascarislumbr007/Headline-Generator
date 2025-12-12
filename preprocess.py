# preprocess.py
import re
import nltk
from nltk.corpus import stopwords
import spacy

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Cleans and normalizes input text for better model input."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\d+", "", text)  
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess_text(text):
    """Performs text normalization using SpaCy."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])
