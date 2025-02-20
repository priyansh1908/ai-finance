import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -------------------- Load Pretrained FinBERT Model --------------------
MODEL_NAME = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Create a sentiment analysis pipeline
finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# -------------------- Save the Pretrained Model --------------------
joblib.dump(finbert_pipeline, "models/sentiment_model.pkl")
print("✅ Pretrained FinBERT Sentiment Model Saved Successfully!")
