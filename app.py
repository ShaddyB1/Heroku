import os
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import logging
from flask.logging import create_logger


warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)
logger = create_logger(app)
logger.setLevel(logging.INFO)

class ParagraphQualityClassifier(nn.Module):
    def __init__(self):
        super(ParagraphQualityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)  

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.dropout(pooled_output)
        return self.fc(x)


current_dir = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(current_dir, 'paragraph_quality_model.pth')


if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = ParagraphQualityClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/classify', methods=['POST'])
def classify_paragraph():
    try:
        data = request.json
        if not data or 'paragraph' not in data:
            logger.warning("No paragraph provided in the request")
            return jsonify({'error': 'No paragraph provided'}), 400

        paragraph = data['paragraph']
        if not paragraph.strip():
            logger.warning("Empty paragraph provided")
            return jsonify({'error': 'Empty paragraph provided'}), 400

        logger.info(f"Classifying paragraph: {paragraph[:50]}...")  # Log first 50 characters

        inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        quality = "High" if predicted_class == 1 else "Low"
        logger.info(f"Classification result: {quality}")
        return jsonify({'quality': quality})
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/', methods=['GET'])
def home():
    logger.info("Received a request to the root route")
    return "Paragraph Quality Classifier API is running!"

if __name__ == '__main__':
    logger.info("Starting the Flask application")
    app.run(debug=True, port=0)  