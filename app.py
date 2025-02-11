import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Initialize the pipeline once at startup
try:
    classifier = pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        device=-1  # Force CPU usage
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return "Paragraph Quality Classifier API"

@app.route('/classify', methods=['POST'])
def classify_paragraph():
    try:
        # Get data from request
        data = request.json
        if not data or 'paragraph' not in data:
            return jsonify({"error": "No paragraph provided"}), 400

        paragraph = data['paragraph']
        
        # Perform classification
        result = classifier(paragraph)[0]
        
        # Convert sentiment to quality rating
        quality = "High" if result['label'] == 'POSITIVE' else "Low"
        confidence = float(result['score'])
        
        # Return result
        return jsonify({
            "quality": quality,
            "confidence": confidence,
            "paragraph": paragraph
        })
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
