"""
Data Preparation Module
This module handles data collection, cleaning, and preprocessing for model training.
"""

import os
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict, load_dataset
import requests
from pathlib import Path

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_sample_data(self):
        """Download sample datasets for demonstration"""
        print("📥 Downloading sample datasets...")
        
        # Load a sample text classification dataset
        try:
            # Using IMDb movie reviews dataset as an example
            dataset = load_dataset("imdb", split="train[:1000]")  # Small subset for demo
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)
            
            # Save raw data
            df.to_csv(self.raw_dir / "imdb_sample.csv", index=False)
            print(f"✅ Downloaded {len(df)} samples to {self.raw_dir / 'imdb_sample.csv'}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            # Create synthetic data as fallback
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic text data for demonstration"""
        print("🔄 Creating synthetic dataset...")
        
        # Sample texts and labels
        texts = [
            "This movie was absolutely fantastic! The acting was superb.",
            "I didn't enjoy this film at all. Very disappointing.",
            "Great storyline and excellent cinematography. Highly recommended!",
            "The plot was confusing and the characters were poorly developed.",
            "One of the best movies I've ever seen. Amazing experience!",
            "Boring and predictable. Waste of time.",
            "Incredible visual effects and outstanding performances.",
            "Not worth watching. Very poor quality.",
            "Brilliant direction and compelling narrative.",
            "Terrible acting and weak script."
        ] * 100  # Multiply to get more samples
        
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100  # 1 = positive, 0 = negative
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Save raw data
        df.to_csv(self.raw_dir / "synthetic_reviews.csv", index=False)
        print(f"✅ Created {len(df)} synthetic samples")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts
        if len(text.split()) < 3:
            return ""
        
        return text
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data"""
        print("🧹 Preprocessing data...")
        
        # Make a copy
        processed_df = df.copy()
        
        # Clean text
        processed_df['text'] = processed_df['text'].apply(self.clean_text)
        
        # Remove empty texts
        processed_df = processed_df[processed_df['text'] != ""]
        
        # Reset index
        processed_df = processed_df.reset_index(drop=True)
        
        # Add text length feature
        processed_df['text_length'] = processed_df['text'].str.len()
        processed_df['word_count'] = processed_df['text'].str.split().str.len()
        
        print(f"✅ Preprocessed data: {len(processed_df)} samples remaining")
        print(f"📊 Average text length: {processed_df['text_length'].mean():.1f} characters")
        print(f"📊 Average word count: {processed_df['word_count'].mean():.1f} words")
        
        return processed_df
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        print("✂️ Splitting data...")
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': df[:train_end],
            'validation': df[train_end:val_end],
            'test': df[val_end:]
        }
        
        for split_name, split_df in splits.items():
            print(f"📊 {split_name}: {len(split_df)} samples")
            
            # Save splits
            split_df.to_csv(self.processed_dir / f"{split_name}.csv", index=False)
        
        return splits
    
    def create_dataset_dict(self, splits: Dict[str, pd.DataFrame]) -> DatasetDict:
        """Convert pandas DataFrames to HuggingFace Dataset format"""
        print("🔄 Converting to HuggingFace Dataset format...")
        
        dataset_dict = {}
        for split_name, df in splits.items():
            # Select only the columns we need for training
            dataset = Dataset.from_pandas(df[['text', 'label']])
            dataset_dict[split_name] = dataset
        
        return DatasetDict(dataset_dict)
    
    def get_data_statistics(self, df: pd.DataFrame):
        """Print data statistics"""
        print("\n📈 Data Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Unique texts: {df['text'].nunique()}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print(f"Text length stats:")
        print(df['text_length'].describe())
    
    def process_pipeline(self) -> DatasetDict:
        """Run the complete data processing pipeline"""
        print("🚀 Starting data processing pipeline...\n")
        
        # Step 1: Download or create data
        raw_data = self.download_sample_data()
        
        # Step 2: Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        # Step 3: Get statistics
        self.get_data_statistics(processed_data)
        
        # Step 4: Split data
        splits = self.split_data(processed_data)
        
        # Step 5: Convert to HuggingFace format
        dataset_dict = self.create_dataset_dict(splits)
        
        # Save the final dataset
        dataset_dict.save_to_disk(str(self.processed_dir / "final_dataset"))
        
        print(f"\n✅ Data processing complete! Dataset saved to {self.processed_dir / 'final_dataset'}")
        
        return dataset_dict

def main():
    """Main function to run data preparation"""
    processor = DataProcessor()
    dataset = processor.process_pipeline()
    
    # Display sample data
    print("\n📋 Sample data:")
    for i in range(3):
        sample = dataset['train'][i]
        print(f"Text: {sample['text'][:100]}...")
        print(f"Label: {sample['label']}\n")

if __name__ == "__main__":
    main()
