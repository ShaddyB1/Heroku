import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import spacy
import re
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model isn't installed, download it
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ParagraphFeatures:
    def __init__(self, text):
        self.text = text
        self.doc = nlp(text)
    
    def get_sentence_count(self):
        return len(list(self.doc.sents))
    
    def get_word_count(self):
        return len([token for token in self.doc if not token.is_punct])
    
    def get_average_sentence_length(self):
        sentences = list(self.doc.sents)
        if not sentences:
            return 0
        return sum(len([token for token in sent if not token.is_punct]) for sent in sentences) / len(sentences)
    
    def get_average_word_length(self):
        words = [token.text for token in self.doc if not token.is_punct and token.is_alpha]
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def get_coherence_score(self):
        sentences = list(self.doc.sents)
        if len(sentences) < 2:
            return 0
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            current_sent = set(token.lemma_ for token in sentences[i] if token.is_alpha)
            next_sent = set(token.lemma_ for token in sentences[i+1] if token.is_alpha)
            if not current_sent or not next_sent:
                continue
            overlap = len(current_sent.intersection(next_sent))
            coherence_scores.append(overlap / max(len(current_sent), len(next_sent)))
        
        return sum(coherence_scores) / max(len(coherence_scores), 1)
    
    def has_topic_sentence(self):
        sentences = list(self.doc.sents)
        if not sentences:
            return 0
        first_sentence = sentences[0]
        return any(token.dep_ == "nsubj" for token in first_sentence)
    
    def get_lexical_diversity(self):
        words = [token.lemma_.lower() for token in self.doc if token.is_alpha]
        if not words:
            return 0
        return len(set(words)) / len(words)
    
    def get_passive_voice_count(self):
        count = 0
        for sent in self.doc.sents:
            for token in sent:
                if token.dep_ == "nsubjpass":
                    count += 1
        return count
    
    def get_punctuation_ratio(self):
        total_tokens = len(self.doc)
        if total_tokens == 0:
            return 0
        punct_count = len([token for token in self.doc if token.is_punct])
        return punct_count / total_tokens
    
    def get_features(self):
        return torch.tensor([
            self.get_sentence_count(),
            self.get_word_count(),
            self.get_average_sentence_length(),
            self.get_average_word_length(),
            self.get_coherence_score(),
            int(self.has_topic_sentence()),
            self.get_lexical_diversity(),
            self.get_passive_voice_count(),
            self.get_punctuation_ratio()
        ], dtype=torch.float)

class ParagraphQualityClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', freeze_bert=False, feature_size=9):
        super(ParagraphQualityClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get the output dimension of the BERT model
        if 'roberta' in model_name:
            bert_output_dim = 768
        elif 'distilbert' in model_name:
            bert_output_dim = 768
        else:  # default for most models
            bert_output_dim = 768
        
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(bert_output_dim + feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take CLS token output
        
        # Combine BERT output with handcrafted features
        combined = torch.cat((pooled_output, features), dim=1)
        
        # Apply layers with batch normalization and dropout
        x = self.fc1(combined)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.fc3(x)

class ParagraphDataset(Dataset):
    def __init__(self, paragraphs, labels, tokenizer, max_length=128):
        self.paragraphs = paragraphs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            paragraph,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        features = ParagraphFeatures(paragraph).get_features()

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': features,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def save_model_artifacts(model, tokenizer, model_path="model_artifacts"):
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(model_path, "tokenizer"))
    
    # Save model config
    model_config = {
        'model_type': model.__class__.__name__,
        'feature_count': 9,  # Current number of features
    }
    
    with open(os.path.join(model_path, "config.json"), 'w') as f:
        json.dump(model_config, f)
    
    print(f"Model artifacts saved to {model_path}")

def load_model_artifacts(model_path="model_artifacts"):
    # Load config
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    
    # Initialize model
    model = ParagraphQualityClassifier(feature_size=config['feature_count'])
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    
    return model, tokenizer

def train_model(model, train_loader, val_loader, epochs=5, lr=2e-5, warmup_steps=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': lr},
        {'params': model.fc1.parameters(), 'lr': lr * 10},
        {'params': model.fc2.parameters(), 'lr': lr * 10},
        {'params': model.fc3.parameters(), 'lr': lr * 10}
    ], lr=lr, weight_decay=0.01)
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # For early stopping
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    best_model_saved = False
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Compute validation metrics
        metrics = evaluate_model(model, val_loader, device)
        history['val_metrics'].append(metrics)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall: {metrics['recall']:.4f}")
        
        # Early stopping
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
            best_model_saved = True
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_saved:
        model.load_state_dict(torch.load('best_model.pt'))
    return model, history

def generate_training_data():
    """Generate a larger dataset for training with more examples"""
    # Check if we have additional training data from external file
    external_data = {"high_quality": [], "medium_quality": [], "low_quality": []}
    external_data_file = "training_data.json"
    
    if os.path.exists(external_data_file):
        try:
            with open(external_data_file, 'r') as f:
                external_data = json.load(f)
                print(f"Loaded additional training data from {external_data_file}")
                print(f"Additional examples: High: {len(external_data['high_quality'])}, " 
                      f"Medium: {len(external_data['medium_quality'])}, "
                      f"Low: {len(external_data['low_quality'])}")
        except Exception as e:
            print(f"Error loading {external_data_file}: {str(e)}")
    
    # High-quality paragraphs (from original dataset)
    high_quality = [
        "This is a well-structured paragraph with clear ideas and proper grammar. It contains multiple sentences that are coherent and related to a single topic. The sentences flow logically from one to the next, creating a unified whole.",
        "The author presents a compelling argument with strong evidence and logical flow. The paragraph begins with a clear topic sentence, followed by supporting details and examples. It concludes with a sentence that reinforces the main idea.",
        "A concise and informative paragraph that effectively communicates its main point. Although brief, it contains multiple sentences that work together to convey a single idea. The language is clear and precise.",
        "This paragraph demonstrates excellent use of vocabulary and varied sentence structure. It begins with a strong topic sentence that introduces the main idea. The following sentences elaborate on this idea, providing specific details and examples. The paragraph concludes with a sentence that ties everything together, reinforcing the central theme.",
        "This well-crafted paragraph showcases the author's command of language. It starts with an engaging topic sentence that captures the reader's attention. The subsequent sentences develop the main idea logically, using specific details and vivid language. The paragraph concludes with a thoughtful statement that leaves a lasting impression on the reader.",
        "The research findings indicate a significant correlation between exercise and mental health. Regular physical activity has been shown to reduce symptoms of depression and anxiety. Additionally, studies have found that even short bursts of exercise can improve mood and cognitive function. These benefits appear to be consistent across different age groups and fitness levels.",
        "Climate change poses substantial threats to coastal communities worldwide. Rising sea levels are already causing increased flooding during high tides and storms. Infrastructure damage is becoming more common, affecting roads, utilities, and buildings. Moreover, saltwater intrusion is contaminating freshwater sources that many communities rely on for drinking and agriculture.",
        "Artificial intelligence has transformed numerous industries over the past decade. Machine learning algorithms now power recommendation systems that personalize our online experiences. Natural language processing enables virtual assistants to understand and respond to human speech with impressive accuracy. Computer vision systems can identify objects and people in images with superhuman precision. These technologies continue to advance at a rapid pace.",
        "Renewable energy adoption has accelerated globally in recent years. Solar panel costs have decreased dramatically, making this technology accessible to more communities. Wind power capacity has expanded significantly, particularly in coastal and plains regions. Innovations in battery storage are addressing intermittency challenges. These developments are crucial for reducing carbon emissions and combating climate change.",
        "Healthy communication forms the foundation of strong relationships. Active listening involves fully focusing on what others are saying rather than preparing your response. Expressing feelings clearly and respectfully prevents misunderstandings from escalating. Regular check-ins create opportunities to address concerns before they become major issues. These practices build trust and mutual understanding over time."
    ]
    
    # Low-quality paragraphs (from original dataset)
    low_quality = [
        "Poor writing confusing ideas no structure. One sentence only.",
        "Lacks coherence jumbled thoughts grammatical errors. Sentences don't connect. Ideas jump around. No clear topic or purpose.",
        "Rambling sentences no clear topic unfocused writing. This text goes on and on without making a point. It's hard to follow because there's no central theme. The ideas are all over the place and don't connect well.",
        "One",
        "The",
        "The quick brown fox jumps over the lazy dog.",
        "Run on sentence with no punctuation or capitalization just a stream of consciousness that goes on and on without any clear breaks or pauses making it difficult to understand or follow the intended meaning if there even is one",
        "Vague generalities without specific examples or supporting evidence. Some say things happen. Others disagree. It's complicated. No one really knows for sure.",
        "Computer fast internet good technology important today modern world changing rapidly innovations happen software hardware devices smart AI learning machine algorithm data big analysis analytics cloud computing systems networks programming languages developers code.",
        "Yesterday went store bought things came home tired watched TV ate dinner slept woke up today same routine boring life continues nothing interesting happened but that's how it goes sometimes nothing to report really just regular day.",
        "School important education valuable learn subjects mathematics science history art music physical education teachers students classroom learning curriculum textbooks homework assignments tests exams grades report cards graduation diplomas certificates.",
        "Cats animals pets furry meow purr sleep laziness play toys mouse catch predator whiskers tail paws fur color breed domestic independent clean litter box.",
        "Weather today sunny rain tomorrow forecast temperature cold warm hot climate change seasons winter summer fall spring predict meteorology clouds sky atmosphere humidity precipitation.",
        "Food delicious tasty cooking recipes ingredients kitchen chef bake fry grill boil steam eat restaurant home meal breakfast lunch dinner snacks appetizers desserts cuisine cultural traditional modern fusion."
    ]
    
    medium_quality = [
        "This paragraph has some good points, but lacks proper structure. The ideas are somewhat related but don't flow naturally from one to the next. There are a few grammatical errors that distract from the message.",
        "The weather has been unusually warm this winter. Many people are enjoying outdoor activities normally reserved for spring. However climate change concerns some scientists.",
        "Technology continues to advance rapidly in modern society. Smartphones apps artificial intelligence changing how we live and work. Many benefits but also some downsides to consider.",
        "Education is important for personal development. Schools provide knowledge and skills. College degrees often lead to better jobs. Alternative paths exist too.",
        "Healthy eating habits contribute to overall wellbeing. Fruits and vegetables contain important nutrients. Processed foods should be limited. Moderation is key to a balanced diet.",
        "Exercise benefits physical and mental health. Regular activity strengthens muscles and cardiovascular system. Endorphins improve mood. Consistency matters more than intensity.",
        "Travel exposes people to new cultures and perspectives. Planning trips can be stressful but rewarding. Budget constraints limit options for many. Virtual tourism offers alternatives.",
        "Reading fiction enhances empathy and vocabulary. Non-fiction provides practical knowledge. Digital books convenient but many prefer physical copies. Libraries remain important community resources."
    ]
    
    # Add user-contributed examples from the external file
    if external_data.get('high_quality'):
        high_quality.extend([item['text'] for item in external_data['high_quality']])
    if external_data.get('medium_quality'):  
        medium_quality.extend([item['text'] for item in external_data['medium_quality']])
    if external_data.get('low_quality'):
        low_quality.extend([item['text'] for item in external_data['low_quality']])
    
    # Combine datasets with labels
    paragraphs = high_quality + medium_quality + low_quality
    # 1 for high quality, 0 for not high quality (medium or low)
    labels = [1] * len(high_quality) + [0] * (len(medium_quality) + len(low_quality))
    
    return paragraphs, labels

def main():
    # Generate training data
    paragraphs, labels = generate_training_data()
    
    print(f"Dataset size: {len(paragraphs)} examples")
    print(f"Positive examples: {sum(labels)}")
    print(f"Negative examples: {len(labels) - sum(labels)}")
    
    # Choose model name
    model_name = 'roberta-base'  # Better than BERT for text classification
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Split data into train, validation, and test sets
    train_paragraphs, temp_paragraphs, train_labels, temp_labels = train_test_split(
        paragraphs, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paragraphs, test_paragraphs, val_labels, test_labels = train_test_split(
        temp_paragraphs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Create datasets
    train_dataset = ParagraphDataset(train_paragraphs, train_labels, tokenizer)
    val_dataset = ParagraphDataset(val_paragraphs, val_labels, tokenizer)
    test_dataset = ParagraphDataset(test_paragraphs, test_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize model
    model = ParagraphQualityClassifier(model_name=model_name)
    
    # Train model
    trained_model, history = train_model(model, train_loader, val_loader, epochs=5)
    
    print("Training completed.")
    
    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = evaluate_model(trained_model, test_loader, device)
    
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save model artifacts
    save_model_artifacts(trained_model, tokenizer)

    # Example usage
    print("\nTesting model on examples:")
    test_examples = [
        "This is a well-written test paragraph to demonstrate the model's capability. It contains multiple sentences that are coherent and related to a single topic. The sentences flow logically, creating a unified whole.",
        "Bad grammar no sense. One sentence only.",
        "One",
        "The quick brown fox jumps over the lazy dog.",
        "This insightful analysis is supported by relevant examples and clear argumentation. The paragraph begins with a strong thesis statement, followed by several sentences that provide evidence and explanation. It concludes by reinforcing the main point, tying all the ideas together effectively."
    ]

    trained_model.eval()
    for test_paragraph in test_examples:
        test_encoding = tokenizer.encode_plus(
            test_paragraph,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        test_features = ParagraphFeatures(test_paragraph).get_features().unsqueeze(0)

        with torch.no_grad():
            outputs = trained_model(
                test_encoding['input_ids'].to(device),
                test_encoding['attention_mask'].to(device),
                test_features.to(device)
            )
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        quality = "High Quality" if predicted_class == 1 else "Low Quality"
        print(f"\nTest Paragraph: {test_paragraph}")
        print(f"Prediction: {quality}")
        print(f"Confidence: {confidence:.2f}")
        
        # Display feature values
        features = ParagraphFeatures(test_paragraph)
        print("Feature values:")
        print(f"  Sentence count: {features.get_sentence_count()}")
        print(f"  Word count: {features.get_word_count()}")
        print(f"  Avg sentence length: {features.get_average_sentence_length():.2f}")
        print(f"  Avg word length: {features.get_average_word_length():.2f}")
        print(f"  Coherence score: {features.get_coherence_score():.2f}")
        print(f"  Has topic sentence: {features.has_topic_sentence()}")
        print(f"  Lexical diversity: {features.get_lexical_diversity():.2f}")
        print(f"  Passive voice count: {features.get_passive_voice_count()}")
        print(f"  Punctuation ratio: {features.get_punctuation_ratio():.2f}")

if __name__ == "__main__":
    main()
