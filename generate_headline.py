# generate_headline.py
import pickle
import nltk
from nltk.tokenize import sent_tokenize
from preprocess import preprocess_text

# Load the saved model & tokenizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def generate_headline(text):
    """Generates short and accurate headlines from news articles."""
    text = preprocess_text(text)
    
    sentences = sent_tokenize(text)
    input_text = "summarize: " + " ".join(sentences)

    # Encode text input
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate headline with optimized parameters
    summary_ids = model.generate(
        inputs, 
        max_length=10,  
        num_beams=8, 
        early_stopping=True, 
        repetition_penalty=2.0, 
        length_penalty=2.5
    )

    headline = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return headline

if __name__ == "__main__":
    sample_text = "Scientists discover a new planet outside the solar system that may have life."
    print("Generated Headline:", generate_headline(sample_text))
