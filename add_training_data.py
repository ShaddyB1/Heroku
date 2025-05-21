#!/usr/bin/env python3
"""
Script to add additional training data to the paragraph quality classifier.
This allows users to easily contribute more examples to improve the model.
"""

import json
import os
import argparse

DATA_FILE = "training_data.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Add training data for paragraph quality classification")
    parser.add_argument("--paragraph", type=str, required=True, 
                        help="The paragraph text to add as a training example")
    parser.add_argument("--quality", type=str, required=True, choices=["high", "low", "medium"], 
                        help="The quality label (high, medium, or low)")
    parser.add_argument("--source", type=str, default="user_contributed", 
                        help="The source of the paragraph (e.g., book, article, user_contributed)")
    return parser.parse_args()

def load_training_data():
    """Load existing training data or create a new file if it doesn't exist"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {DATA_FILE}, creating a new file.")
                return {"high_quality": [], "medium_quality": [], "low_quality": []}
    else:
        return {"high_quality": [], "medium_quality": [], "low_quality": []}

def save_training_data(data):
    """Save training data to file"""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def add_example(paragraph, quality, source):
    """Add a new example to the training data"""
    # Load existing data
    training_data = load_training_data()
    
    # Map quality to the relevant key
    quality_key = f"{quality}_quality"
    
    if quality_key not in training_data:
        print(f"Invalid quality: {quality}. Must be 'high', 'medium', or 'low'.")
        return False
    
    # Create example record
    example = {
        "text": paragraph,
        "source": source,
        "added_at": str(pd.Timestamp.now())
    }
    
    # Add to appropriate list
    training_data[quality_key].append(example)
    
    # Save updated data
    save_training_data(training_data)
    
    print(f"Added new {quality} quality example to {DATA_FILE}")
    print(f"Current counts: High: {len(training_data['high_quality'])}, "
          f"Medium: {len(training_data['medium_quality'])}, "
          f"Low: {len(training_data['low_quality'])}")
    
    return True

def main():
    args = parse_args()
    add_example(args.paragraph, args.quality, args.source)
    print("To retrain the model with this new data, run: python train_model.py")

if __name__ == "__main__":
    import pandas as pd  # Only used for timestamp
    main() 