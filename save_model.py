import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "t5-base"

# Load the model and tokenizer from Hugging Face
print("Downloading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully as model.pkl!")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved successfully as tokenizer.pkl!")
