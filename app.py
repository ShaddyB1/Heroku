import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load model and tokenizer"""
    global model, tokenizer
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Ensure model is in evaluation mode
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.before_first_request
def initialize():
    """Initialize model before first request"""
    load_model()

@app.route('/')
def home():
    return "Paragraph Quality Classifier API"

@app.route('/classify', methods=['POST'])
def classify_paragraph():
    global model, tokenizer
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        if not load_model():
            return jsonify({"error": "Model not initialized"}), 500

    try:
        # Get and validate input
        data = request.get_json()
        if not data or 'paragraph' not in data:
            return jsonify({"error": "No paragraph provided"}), 400

        paragraph = data['paragraph']
        
        # Tokenize and prepare input
        inputs = tokenizer(paragraph, 
                         return_tensors="pt", 
                         truncation=True, 
                         padding=True, 
                         max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities).item()
        
        # Prepare response
        quality = "High" if predicted_class == 1 else "Low"
        confidence = float(probabilities[0][predicted_class].item())
        
        return jsonify({
            "quality": quality,
            "confidence": confidence,
            "paragraph": paragraph
        })

    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the port provided by Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    
