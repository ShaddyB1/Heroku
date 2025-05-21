# Advanced Paragraph Quality Analyzer

An AI-powered tool for analyzing and improving written paragraphs using Natural Language Processing and Machine Learning.

## Overview

The Advanced Paragraph Quality Analyzer helps writers evaluate the quality of their paragraphs by analyzing various linguistic features. It provides detailed feedback and suggestions for improvement based on an advanced machine learning model.

![Screenshot of the application](https://placeholder-image.com/screenshot.png)

## Features

- **Quality Assessment**: Determines if a paragraph is high or low quality
- **Confidence Score**: Shows how confident the model is in its assessment
- **Detailed Feature Analysis**: Analyzes and displays 9 linguistic features:
  - Sentence count
  - Word count
  - Average sentence length
  - Average word length
  - Coherence score
  - Topic sentence presence
  - Lexical diversity
  - Passive voice count
  - Punctuation ratio
- **Improvement Suggestions**: Provides customized suggestions based on the analysis

## Technology Stack

- **Backend**: Python, Flask, PyTorch, Transformers (RoBERTa)
- **NLP Processing**: spaCy
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Heroku

## How It Works

1. The tool uses a fine-tuned RoBERTa language model combined with handcrafted linguistic features
2. It analyzes paragraphs based on both semantic understanding and structural characteristics
3. The model provides a binary classification (high/low quality) with a confidence score
4. Additional analysis generates specific improvement suggestions

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/paragraph-quality-analyzer.git
cd paragraph-quality-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Train the model (optional - pre-trained model is included):
```bash
python train_model.py --epochs 5 --visualize
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:10000
```

## Usage

1. Enter or paste your paragraph in the text area
2. Click "Analyze Paragraph"
3. Review the quality assessment, feature analysis, and improvement suggestions
4. Make improvements to your paragraph based on the feedback
5. Re-analyze to see if the quality has improved

## Model Training

The model can be retrained using the provided script:

```bash
python train_model.py --model roberta-base --epochs 5 --batch-size 8 --lr 2e-5 --visualize
```

Available options:
- `--model`: Transformer model to use (default: roberta-base)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 2e-5)
- `--output-dir`: Directory to save model artifacts (default: model_artifacts)
- `--visualize`: Generate training visualizations

## Deployment

The application is configured for easy deployment to Heroku:

```bash
git push heroku main
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The transformers library from Hugging Face
- spaCy for natural language processing
- The open-source NLP community 