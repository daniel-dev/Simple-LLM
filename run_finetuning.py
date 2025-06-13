"""
Fine-tuning Demonstration Script
"""

import os
import sys
sys.path.append('src')

from fine_tuning import AdvancedFineTuner
from pathlib import Path

def main():
    """Run LoRA fine-tuning demonstration"""
    print("🔧 Starting LoRA Fine-tuning Demonstration...")
    
    # Check if base model exists
    base_model_path = "models/trained/simple_bert"
    if not Path(base_model_path).exists():
        print("❌ Base model not found. Please run training first.")
        return
    
    try:
        # Initialize fine-tuner
        fine_tuner = AdvancedFineTuner(
            base_model_path=base_model_path,
            output_dir="models/fine_tuned/lora_bert",
            use_lora=True
        )
        
        # Setup LoRA configuration
        fine_tuner.setup_lora_config(
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        # Check for tokenized dataset
        tokenized_path = "data/processed/tokenized_dataset"
        if not Path(tokenized_path).exists():
            print("❌ Tokenized dataset not found. Please run tokenization first.")
            return
        
        # Run LoRA fine-tuning
        print("🚀 Starting LoRA fine-tuning...")
        fine_tuner.fine_tune_with_lora(
            tokenized_path,
            num_epochs=1,  # Quick demo
            batch_size=4
        )
        
        print("✅ LoRA fine-tuning completed successfully!")
        
        # Test the fine-tuned model
        print("🔍 Testing fine-tuned model...")
        fine_tuner.test_model()
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
