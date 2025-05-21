#!/usr/bin/env python3
"""
Paragraph Quality Classifier - Training Script
Trains and saves the model for the paragraph quality app.
"""

import os
import argparse
from paragraph_quality import (
    generate_training_data, 
    ParagraphDataset, 
    ParagraphQualityClassifier,
    train_model,
    evaluate_model,
    save_model_artifacts
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train the paragraph quality classifier model")
    parser.add_argument("--model", type=str, default="roberta-base", 
                        help="Transformer model to use (default: roberta-base)")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size for training (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--output-dir", type=str, default="model_artifacts", 
                        help="Directory to save model artifacts (default: model_artifacts)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate training visualizations")
    return parser.parse_args()

def visualize_training(history, output_dir):
    """Generate visualizations of training metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    # Plot validation metrics over time
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        values = [epoch_metrics[metric] for epoch_metrics in history['val_metrics']]
        plt.subplot(2, 2, i+1)
        plt.plot(values, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'Validation {metric.capitalize()}')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    
    # Plot confusion matrix for the last epoch
    last_cm = history['val_metrics'][-1]['confusion_matrix']
    plt.figure(figsize=(8, 6))
    plt.imshow(last_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Low Quality', 'High Quality']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = last_cm.max() / 2.
    for i in range(last_cm.shape[0]):
        for j in range(last_cm.shape[1]):
            plt.text(j, i, format(last_cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if last_cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    print(f"Training visualizations saved to {output_dir}")

def main():
    args = parse_args()
    
    print(f"=== Training Paragraph Quality Classifier ===")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Visualize: {args.visualize}")
    print("="*42)
    
    # Generate training data
    paragraphs, labels = generate_training_data()
    
    print(f"Dataset size: {len(paragraphs)} examples")
    print(f"Positive examples: {sum(labels)}")
    print(f"Negative examples: {len(labels) - sum(labels)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = ParagraphQualityClassifier(model_name=args.model)
    
    # Train model
    print("\nStarting training...\n")
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        lr=args.lr
    )
    
    print("\nTraining completed.")
    
    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_metrics = evaluate_model(trained_model, test_loader, device)
    
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Save model artifacts
    save_model_artifacts(trained_model, tokenizer, args.output_dir)
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating training visualizations...")
        visualize_training(history, args.output_dir)
    
    print(f"\nModel artifacts saved to {args.output_dir}")
    print("\nTo use this model in the web app, start the app with 'python app.py'")

if __name__ == "__main__":
    main() 