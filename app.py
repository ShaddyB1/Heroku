import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
from paragraph_quality import ParagraphFeatures, ParagraphQualityClassifier, load_model_artifacts

app = Flask(__name__, static_folder='.')
CORS(app)

# Model paths
MODEL_DIR = "model_artifacts"

# Load model and tokenizer at startup
print("Loading model and tokenizer...")
try:
    model, tokenizer = load_model_artifacts(MODEL_DIR)
    model.eval()  # Set model to evaluation mode
    print("Model and tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Using default model as fallback...")
    # If not found, will create a temp model just to make the app work
    # This will be replaced when we run the training script
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

# Map to store device for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/classify', methods=['POST'])
def classify_paragraph():
    try:
        # Get and validate input
        data = request.get_json()
        if not data or 'paragraph' not in data:
            return jsonify({"error": "No paragraph provided"}), 400

        paragraph = data['paragraph']
        
        # Get feature details for the response
        features = ParagraphFeatures(paragraph)
        feature_values = {
            "sentence_count": int(features.get_sentence_count()),
            "word_count": int(features.get_word_count()),
            "avg_sentence_length": float(features.get_average_sentence_length()),
            "avg_word_length": float(features.get_average_word_length()),
            "coherence_score": float(features.get_coherence_score()),
            "has_topic_sentence": bool(features.has_topic_sentence()),
            "lexical_diversity": float(features.get_lexical_diversity()),
            "passive_voice_count": int(features.get_passive_voice_count()),
            "punctuation_ratio": float(features.get_punctuation_ratio())
        }
        
        # Prepare model inputs
        encoding = tokenizer.encode_plus(
            paragraph,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        feature_tensor = features.get_features().unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            # Check if we're using our custom model or the fallback model
            if hasattr(model, 'bert') and hasattr(model, 'fc1'): 
                # Our custom model
                outputs = model(
                    encoding['input_ids'].to(device),
                    encoding['attention_mask'].to(device),
                    feature_tensor.to(device)
                )
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = float(probabilities[0][predicted_class].item())
                quality = "High" if predicted_class == 1 else "Low"
            else:
                # Fallback model
                outputs = model(**{k: v.to(device) for k, v in encoding.items()})
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities).item()
                confidence = float(probabilities[0][predicted_class].item())
                quality = "High" if predicted_class == 1 else "Low"
        
        # Get improvement suggestions
        suggestions = generate_improvement_suggestions(paragraph, feature_values, quality)
        
        # Prepare response
        return jsonify({
            "quality": quality,
            "confidence": confidence,
            "paragraph": paragraph,
            "features": feature_values,
            "suggestions": suggestions
        })

    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_improvement_suggestions(paragraph, features, quality):
    """Generate improvement suggestions based on features"""
    suggestions = []
    
    # Suggestions based on length
    if features["sentence_count"] < 2:
        suggestions.append("Add more sentences to develop your ideas fully.")
    
    if features["word_count"] < 10:
        suggestions.append("Your paragraph is very short. Consider expanding it with more details.")
    
    # Suggestions based on structure
    if not features["has_topic_sentence"]:
        suggestions.append("Consider adding a clear topic sentence at the beginning of your paragraph.")
    
    # Suggestions based on coherence
    if features["coherence_score"] < 0.2 and features["sentence_count"] > 1:
        suggestions.append("Improve the connection between sentences by using transition words and related terms.")
    
    # Suggestions based on lexical diversity
    if features["lexical_diversity"] < 0.4 and features["word_count"] > 20:
        suggestions.append("Try using a wider variety of words to make your writing more engaging.")
    
    # Suggestions based on sentence length
    if features["avg_sentence_length"] > 30:
        suggestions.append("Consider breaking up longer sentences for better readability.")
    elif features["avg_sentence_length"] < 5 and features["sentence_count"] > 1:
        suggestions.append("Try combining some short sentences or adding more detail to them.")
    
    # If no specific suggestions and quality is low
    if not suggestions and quality == "Low":
        suggestions.append("Try to develop a clear main idea with supporting details.")
    
    # If it's a good paragraph but could still improve
    if quality == "High" and not suggestions:
        suggestions.append("Your paragraph is well-structured! For further improvement, consider adding more specific examples or evidence.")
    
    return suggestions

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
