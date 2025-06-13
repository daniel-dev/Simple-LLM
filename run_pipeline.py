"""
Main Pipeline Script
Run the complete machine learning pipeline from data to distilled models
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from data_preparation import DataProcessor
from tokenization import TextTokenizer
from model_training import ModelTrainer
from fine_tuning import AdvancedFineTuner
from distillation import KnowledgeDistiller

def run_complete_pipeline(
    use_existing_data: bool = False,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 2,
    batch_size: int = 8,
    enable_fine_tuning: bool = True,
    enable_distillation: bool = True
):
    """
    Run the complete ML pipeline
    
    Args:
        use_existing_data: Whether to use existing processed data
        model_name: Base model to use
        num_epochs: Number of training epochs
        batch_size: Training batch size
        enable_fine_tuning: Whether to run fine-tuning step
        enable_distillation: Whether to run distillation step
    """
    
    print("🚀 Starting Complete ML Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Data Preparation
    print("\n📊 STEP 1: DATA PREPARATION")
    print("-" * 40)
    
    if not use_existing_data or not Path("data/processed/final_dataset").exists():
        processor = DataProcessor()
        dataset = processor.process_pipeline()
    else:
        print("📂 Using existing processed dataset")
        from datasets import load_from_disk
        dataset = load_from_disk("data/processed/final_dataset")
    
    # Step 2: Tokenization
    print("\n🔤 STEP 2: TOKENIZATION")
    print("-" * 40)
    
    if not Path("data/processed/tokenized_dataset").exists():
        tokenizer = TextTokenizer(model_name=model_name, max_length=128)
        tokenized_dataset = tokenizer.tokenize_dataset(dataset)
        tokenized_dataset.save_to_disk("data/processed/tokenized_dataset")
    else:
        print("📂 Using existing tokenized dataset")
        from datasets import load_from_disk
        tokenized_dataset = load_from_disk("data/processed/tokenized_dataset")
    
    # Step 3: Model Training
    print("\n🤖 STEP 3: MODEL TRAINING")
    print("-" * 40)
    
    if not Path("models/trained/bert_classifier").exists():
        trainer = ModelTrainer(
            model_name=model_name,
            num_labels=2,
            output_dir="models/trained/bert_classifier"
        )
        
        trained_model, train_results = trainer.train_model(
            tokenized_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=2e-5
        )
        
        print(f"✅ Training completed! Accuracy: {train_results['test_results']['eval_accuracy']:.3f}")
    else:
        print("📂 Using existing trained model")
    
    # Step 4: Fine-tuning (Optional)
    if enable_fine_tuning:
        print("\n🔧 STEP 4: FINE-TUNING")
        print("-" * 40)
        
        fine_tuner = AdvancedFineTuner(
            base_model_path="models/trained/bert_classifier",
            output_dir="models/fine_tuned"
        )
        
        # Run LoRA fine-tuning only for speed
        fine_tuner.load_base_model()
        lora_trainer, lora_train, lora_eval = fine_tuner.fine_tune_with_lora(
            tokenized_dataset,
            num_epochs=2,
            batch_size=batch_size,
            learning_rate=1e-4
        )
        
        print(f"✅ LoRA fine-tuning completed! Accuracy: {lora_eval['eval_accuracy']:.3f}")
    else:
        print("⏭️ Skipping fine-tuning step")
    
    # Step 5: Knowledge Distillation (Optional)
    if enable_distillation:
        print("\n🧠 STEP 5: KNOWLEDGE DISTILLATION")
        print("-" * 40)
        
        distiller = KnowledgeDistiller(
            teacher_model_path="models/trained/bert_classifier",
            student_model_name="distilbert-base-uncased",
            output_dir="models/distilled/distilbert_student"
        )
        
        distilled_trainer, distill_results = distiller.distill_model(
            tokenized_dataset,
            num_epochs=3,
            batch_size=batch_size,
            learning_rate=5e-5
        )
        
        # Compare models
        teacher_results, student_results = distiller.compare_models(tokenized_dataset)
        
        print(f"✅ Distillation completed!")
        print(f"📊 Student Accuracy: {student_results['accuracy']:.3f}")
        print(f"📦 Compression: {distill_results['compression_ratio']:.1%} of original size")
    else:
        print("⏭️ Skipping distillation step")
    
    # Pipeline Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏱️ Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\n📁 Generated Files:")
    print("  - data/processed/final_dataset (processed data)")
    print("  - data/processed/tokenized_dataset (tokenized data)")
    print("  - models/trained/bert_classifier (trained model)")
    
    if enable_fine_tuning:
        print("  - models/fine_tuned/lora_adapter (LoRA adapter)")
    
    if enable_distillation:
        print("  - models/distilled/distilbert_student (distilled model)")
    
    print("\n🎯 Pipeline Steps Completed:")
    print("  ✅ Data Preparation")
    print("  ✅ Tokenization")
    print("  ✅ Model Training")
    if enable_fine_tuning:
        print("  ✅ Fine-tuning (LoRA)")
    if enable_distillation:
        print("  ✅ Knowledge Distillation")
    
    print("\n📖 Next Steps:")
    print("  1. Open notebooks/complete_pipeline.ipynb for interactive exploration")
    print("  2. Test your models with new data")
    print("  3. Deploy models for production use")
    print("  4. Experiment with different architectures and techniques")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Complete ML Pipeline")
    
    parser.add_argument("--use-existing-data", action="store_true",
                        help="Use existing processed data if available")
    parser.add_argument("--model-name", default="bert-base-uncased",
                        help="Base model name (default: bert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--skip-fine-tuning", action="store_true",
                        help="Skip fine-tuning step")
    parser.add_argument("--skip-distillation", action="store_true",
                        help="Skip distillation step")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with minimal epochs and small batches")
    
    args = parser.parse_args()
    
    # Quick mode settings
    if args.quick:
        args.epochs = 1
        args.batch_size = 4
        print("🏃‍♂️ Quick mode enabled: 1 epoch, batch size 4")
    
    # Run pipeline
    run_complete_pipeline(
        use_existing_data=args.use_existing_data,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        enable_fine_tuning=not args.skip_fine_tuning,
        enable_distillation=not args.skip_distillation
    )

if __name__ == "__main__":
    main()
