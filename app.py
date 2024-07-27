import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/')
def home():
    return "Paragraph Quality Classifier API"

@app.route('/classify', methods=['POST'])
def classify_paragraph():
    data = request.json
    paragraph = data['paragraph']
    
    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    
    quality = "High" if predicted_class == 1 else "Low"
    confidence = probabilities[0][predicted_class].item()
    
    return jsonify({"quality": quality, "confidence": confidence})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)