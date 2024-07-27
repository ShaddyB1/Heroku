import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paragraph quality classifier model
class ParagraphQualityClassifier(nn.Module):
    def __init__(self):
        super(ParagraphQualityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)  # 2 output classes: low quality (0) and high quality (1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.dropout(pooled_output)
        return self.fc(x)

# Dataset class for paragraphs
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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=5, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        print()

    return model

# Main function
def main():
    # Sample paragraphs and labels
    paragraphs = [
        "This is a well-structured paragraph with clear ideas and proper grammar.",
        "Poor writing confusing ideas no structure.",
        "The author presents a compelling argument with strong evidence and logical flow.",
        "Lacks coherence jumbled thoughts grammatical errors.",
        "A concise and informative paragraph that effectively communicates its main point.",
        "Rambling sentences no clear topic unfocused writing.",
        "The paragraph demonstrates excellent use of vocabulary and varied sentence structure.",
        "Repetitive ideas poor word choice lack of transitions.",
        "A balanced and nuanced discussion of a complex topic in a single paragraph.",
        "Overly simplistic analysis fails to address key points."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for high quality, 0 for low quality

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split data into train and validation sets
    train_paragraphs, val_paragraphs, train_labels, val_labels = train_test_split(
        paragraphs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = ParagraphDataset(train_paragraphs, train_labels, tokenizer)
    val_dataset = ParagraphDataset(val_paragraphs, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize model
    model = ParagraphQualityClassifier()

    # Train model
    trained_model = train_model(model, train_loader, val_loader, epochs=5)

    print("Training completed.")

    # Save the trained model
    torch.save(trained_model.state_dict(), 'paragraph_quality_model.pth')
    print("Model saved successfully.")

    # Test the model with a new paragraph
    test_paragraph = "This is a test paragraph to demonstrate the model's capability."
    test_encoding = tokenizer.encode_plus(
        test_paragraph,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(test_encoding['input_ids'], test_encoding['attention_mask'])
        _, predicted = torch.max(outputs, 1)

    quality = "High" if predicted.item() == 1 else "Low"
    print(f"\nTest Paragraph: {test_paragraph}")
    print(f"Predicted Quality: {quality}")

if __name__ == "__main__":
    main()