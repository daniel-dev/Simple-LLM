"""
Tokenization Module
This module handles text tokenization for model training.
"""

import os
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    PreTrainedTokenizer
)
from datasets import Dataset, DatasetDict
from pathlib import Path

class TextTokenizer:
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 cache_dir: str = "models/tokenizers"):
        """
        Initialize the tokenizer
        
        Args:
            model_name: Name of the pre-trained model/tokenizer
            max_length: Maximum sequence length
            cache_dir: Directory to cache tokenizers
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        print(f"🔤 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize text examples
        
        Args:
            examples: Dictionary containing 'text' key with list of texts
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None  # Return lists, not tensors for datasets
        )
    
    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Tokenize entire dataset
        
        Args:
            dataset: HuggingFace DatasetDict
            
        Returns:
            Tokenized DatasetDict
        """
        print("🔤 Tokenizing dataset...")
        
        # Tokenize all splits
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        # Add back the labels
        def add_labels(examples, indices):
            # Get original labels
            original_examples = dataset['train'].select(indices) if 'train' in dataset else dataset['validation'].select(indices)
            examples['labels'] = original_examples['label']
            return examples
        
        # Process each split
        final_dataset = {}
        for split_name, split_data in tokenized_dataset.items():
            print(f"📊 Processing {split_name} split: {len(split_data)} samples")
            
            # Get original split
            original_split = dataset[split_name]
            
            # Add labels back
            tokenized_with_labels = []
            for i, tokenized_example in enumerate(split_data):
                tokenized_example['labels'] = original_split[i]['label']
                tokenized_with_labels.append(tokenized_example)
            
            # Convert back to dataset
            final_dataset[split_name] = Dataset.from_list(tokenized_with_labels)
        
        tokenized_dataset = DatasetDict(final_dataset)
        
        # Print tokenization statistics
        self.print_tokenization_stats(tokenized_dataset)
        
        return tokenized_dataset
    
    def print_tokenization_stats(self, tokenized_dataset: DatasetDict):
        """Print tokenization statistics"""
        print("\n📊 Tokenization Statistics:")
        
        for split_name, split_data in tokenized_dataset.items():
            sample = split_data[0]
            print(f"\n{split_name.upper()} Split:")
            print(f"  - Samples: {len(split_data)}")
            print(f"  - Features: {list(sample.keys())}")
            print(f"  - Input IDs shape: {len(sample['input_ids'])}")
            print(f"  - Attention mask shape: {len(sample['attention_mask'])}")
            
            # Check sequence lengths
            lengths = [len(example['input_ids']) for example in split_data]
            print(f"  - Sequence length stats:")
            print(f"    * Min: {min(lengths)}")
            print(f"    * Max: {max(lengths)}")
            print(f"    * Average: {sum(lengths)/len(lengths):.1f}")
    
    def create_data_collator(self):
        """Create data collator for dynamic padding"""
        return DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
    
    def decode_example(self, tokenized_example: Dict, show_special_tokens: bool = False):
        """
        Decode a tokenized example back to text
        
        Args:
            tokenized_example: Dictionary with 'input_ids'
            show_special_tokens: Whether to show special tokens
        """
        decoded_text = self.tokenizer.decode(
            tokenized_example['input_ids'],
            skip_special_tokens=not show_special_tokens
        )
        return decoded_text
    
    def analyze_vocabulary(self, dataset: DatasetDict):
        """Analyze vocabulary coverage"""
        print("\n📚 Vocabulary Analysis:")
        
        # Get vocabulary size
        vocab_size = len(self.tokenizer.vocab)
        print(f"Tokenizer vocabulary size: {vocab_size:,}")
        
        # Analyze a sample of texts
        sample_texts = []
        for split_data in dataset.values():
            sample_texts.extend([example['text'] for example in split_data.select(range(min(100, len(split_data))))])
        
        # Tokenize samples
        sample_tokens = []
        for text in sample_texts:
            tokens = self.tokenizer.tokenize(text)
            sample_tokens.extend(tokens)
        
        # Count unique tokens
        unique_tokens = set(sample_tokens)
        print(f"Unique tokens in sample: {len(unique_tokens):,}")
        print(f"Total tokens in sample: {len(sample_tokens):,}")
        print(f"Average tokens per text: {len(sample_tokens)/len(sample_texts):.1f}")
        
        # Show some example tokens
        print(f"Sample tokens: {list(unique_tokens)[:20]}")
    
    def save_tokenizer(self, save_path: str):
        """Save tokenizer to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save_pretrained(save_path)
        print(f"💾 Tokenizer saved to {save_path}")
    
    def example_tokenization(self, texts: List[str]):
        """Show example tokenization"""
        print("\n🔍 Example Tokenization:")
        
        for i, text in enumerate(texts[:3]):
            print(f"\nExample {i+1}:")
            print(f"Original: {text}")
            
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            print(f"Tokens: {tokens}")
            
            # Get input IDs
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            print(f"Input IDs: {encoded}")
            
            # Decode back
            decoded = self.tokenizer.decode(encoded)
            print(f"Decoded: {decoded}")

def main():
    """Main function to run tokenization"""
    from datasets import load_from_disk
    
    # Load processed dataset
    try:
        dataset = load_from_disk("data/processed/final_dataset")
        print("✅ Loaded processed dataset")
    except:
        print("❌ No processed dataset found. Please run data_preparation.py first.")
        return
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(
        model_name="bert-base-uncased",
        max_length=128  # Smaller for demo
    )
    
    # Analyze vocabulary
    tokenizer.analyze_vocabulary(dataset)
    
    # Show example tokenization
    sample_texts = [dataset['train'][i]['text'] for i in range(3)]
    tokenizer.example_tokenization(sample_texts)
    
    # Tokenize dataset
    tokenized_dataset = tokenizer.tokenize_dataset(dataset)
    
    # Save tokenized dataset
    tokenized_dataset.save_to_disk("data/processed/tokenized_dataset")
    
    # Save tokenizer
    tokenizer.save_tokenizer("models/tokenizers/bert_tokenizer")
    
    # Create data collator
    data_collator = tokenizer.create_data_collator()
    
    print("\n✅ Tokenization complete!")
    print(f"📁 Tokenized dataset saved to: data/processed/tokenized_dataset")
    print(f"🔤 Tokenizer saved to: models/tokenizers/bert_tokenizer")

if __name__ == "__main__":
    main()
