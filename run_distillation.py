"""
Knowledge Distillation Demonstration Script
"""

import os
import sys
sys.path.append('src')

from distillation import KnowledgeDistiller
from pathlib import Path

def main():
    """Run knowledge distillation demonstration"""
    print("🧠 Starting Knowledge Distillation Demonstration...")
    
    # Check if teacher model exists
    teacher_model_path = "models/trained/simple_bert"
    if not Path(teacher_model_path).exists():
        print("❌ Teacher model not found. Please run training first.")
        return
    
    try:
        # Initialize distiller
        distiller = KnowledgeDistiller(
            teacher_model_path=teacher_model_path,
            student_model_name="distilbert-base-uncased",
            output_dir="models/distilled/student_bert"
        )
        
        # Check for tokenized dataset
        tokenized_path = "data/processed/tokenized_dataset"
        if not Path(tokenized_path).exists():
            print("❌ Tokenized dataset not found. Please run tokenization first.")
            return
        
        # Run distillation
        print("🔬 Starting knowledge distillation...")
        distiller.distill_knowledge(
            tokenized_path,
            num_epochs=1,  # Quick demo
            batch_size=4,
            temperature=3.0,
            alpha=0.7
        )
        
        print("✅ Knowledge distillation completed successfully!")
        
        # Compare models
        print("📊 Comparing teacher vs student models...")
        distiller.compare_models()
        
    except Exception as e:
        print(f"❌ Error during distillation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
