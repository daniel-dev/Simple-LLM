"""
Fine-tuning Module
This module handles advanced fine-tuning techniques including LoRA and other PEFT methods.
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import load_from_disk
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class AdvancedFineTuner:
    def __init__(self, 
                 base_model_path: str,
                 output_dir: str = "models/fine_tuned",
                 use_lora: bool = True):
        """
        Initialize advanced fine-tuner
        
        Args:
            base_model_path: Path to the base trained model
            output_dir: Directory to save fine-tuned models
            use_lora: Whether to use LoRA (Low-Rank Adaptation)
        """
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_lora = use_lora
        
        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_base_model()
        
    def load_base_model(self):
        """Load the base model and tokenizer"""
        print(f"📂 Loading base model from: {self.base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"✅ Base model loaded with {self.model.num_parameters():,} parameters")
    
    def setup_lora_config(self,
                         r: int = 16,
                         lora_alpha: int = 32,
                         target_modules: list = None,
                         lora_dropout: float = 0.1):
        """
        Setup LoRA configuration
        
        Args:
            r: Rank of adaptation
            lora_alpha: LoRA scaling parameter
            target_modules: Target modules to apply LoRA
            lora_dropout: LoRA dropout
        """
        if target_modules is None:
            # Common target modules for BERT-like models
            target_modules = ["query", "value", "key", "dense"]
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            inference_mode=False
        )
        
        return lora_config
    
    def apply_lora(self, lora_config):
        """Apply LoRA to the model"""
        print("🔧 Applying LoRA adaptation...")
        
        # Prepare model for k-bit training if using quantization
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print("✅ LoRA applied successfully")
        
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fine_tune_with_lora(self,
                           tokenized_dataset,
                           num_epochs: int = 3,
                           batch_size: int = 8,
                           learning_rate: float = 1e-4,
                           lora_r: int = 16):
        """
        Fine-tune model using LoRA
        
        Args:
            tokenized_dataset: Tokenized dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            lora_r: LoRA rank
        """
        print("🚀 Starting LoRA fine-tuning...")
        
        # Setup LoRA
        lora_config = self.setup_lora_config(r=lora_r)
        self.model = self.apply_lora(lora_config)
        
        # Create training arguments        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "lora"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            eval_strategy="steps",  # Updated from evaluation_strategy
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate(tokenized_dataset["test"])
        
        # Save LoRA adapter
        self.model.save_pretrained(str(self.output_dir / "lora_adapter"))
        
        print("📊 LoRA Fine-tuning Results:")
        print(f"  - Train Loss: {train_result.training_loss:.4f}")
        print(f"  - Test Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"  - Test F1: {eval_result['eval_f1']:.4f}")
        
        return trainer, train_result, eval_result
    
    def full_fine_tune(self,
                      tokenized_dataset,
                      num_epochs: int = 2,
                      batch_size: int = 4,
                      learning_rate: float = 5e-6):
        """
        Full fine-tuning (all parameters)
        
        Args:
            tokenized_dataset: Tokenized dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate (lower for full fine-tuning)
        """
        print("🚀 Starting full fine-tuning...")
        
        # Create training arguments with lower learning rate
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "full_finetune"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,
            gradient_checkpointing=True,  # Save memory
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate(tokenized_dataset["test"])
        
        # Save model
        trainer.save_model()
        
        print("📊 Full Fine-tuning Results:")
        print(f"  - Train Loss: {train_result.training_loss:.4f}")
        print(f"  - Test Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"  - Test F1: {eval_result['eval_f1']:.4f}")
        
        return trainer, train_result, eval_result
    
    def compare_approaches(self, tokenized_dataset):
        """Compare different fine-tuning approaches"""
        print("🔄 Comparing fine-tuning approaches...")
        
        results = {}
        
        # 1. LoRA Fine-tuning
        print("\n" + "="*50)
        print("1. LoRA FINE-TUNING")
        print("="*50)
        
        # Reset model for LoRA
        self.load_base_model()
        lora_trainer, lora_train, lora_eval = self.fine_tune_with_lora(
            tokenized_dataset,
            num_epochs=2,
            batch_size=8,
            learning_rate=1e-4
        )
        
        results['lora'] = {
            'train_loss': lora_train.training_loss,
            'test_accuracy': lora_eval['eval_accuracy'],
            'test_f1': lora_eval['eval_f1'],
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # 2. Full Fine-tuning (if computational resources allow)
        print("\n" + "="*50)
        print("2. FULL FINE-TUNING")
        print("="*50)
        
        # Reset model for full fine-tuning
        self.load_base_model()
        full_trainer, full_train, full_eval = self.full_fine_tune(
            tokenized_dataset,
            num_epochs=1,  # Fewer epochs for demo
            batch_size=4,   # Smaller batch for memory
            learning_rate=5e-6
        )
        
        results['full'] = {
            'train_loss': full_train.training_loss,
            'test_accuracy': full_eval['eval_accuracy'],
            'test_f1': full_eval['eval_f1'],
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Print comparison
        self.print_comparison(results)
        
        return results
    
    def print_comparison(self, results):
        """Print comparison of different approaches"""
        print("\n" + "="*60)
        print("📊 FINE-TUNING APPROACHES COMPARISON")
        print("="*60)
        
        print(f"{'Method':<15} {'Accuracy':<10} {'F1':<10} {'Trainable Params':<15}")
        print("-" * 60)
        
        for method, metrics in results.items():
            print(f"{method.upper():<15} "
                  f"{metrics['test_accuracy']:<10.4f} "
                  f"{metrics['test_f1']:<10.4f} "
                  f"{metrics['trainable_params']:,}")
        
        # Calculate efficiency
        if 'lora' in results and 'full' in results:
            param_ratio = results['lora']['trainable_params'] / results['full']['trainable_params']
            acc_ratio = results['lora']['test_accuracy'] / results['full']['test_accuracy']
            
            print(f"\n💡 LoRA Efficiency:")
            print(f"  - Uses {param_ratio:.1%} of trainable parameters")
            print(f"  - Achieves {acc_ratio:.1%} of full fine-tuning accuracy")
    
    def load_lora_model(self, lora_adapter_path: str, base_model_path: str = None):
        """Load a LoRA adapter"""
        if base_model_path is None:
            base_model_path = self.base_model_path
        
        print(f"📂 Loading LoRA adapter from: {lora_adapter_path}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        return model

def main():
    """Main function to run fine-tuning"""
    # Load tokenized dataset
    try:
        tokenized_dataset = load_from_disk("data/processed/tokenized_dataset")
        print("✅ Loaded tokenized dataset")
    except:
        print("❌ No tokenized dataset found. Please run previous steps first.")
        return
    
    # Check if base model exists
    base_model_path = "models/trained/bert_classifier"
    if not Path(base_model_path).exists():
        print("❌ No base trained model found. Please run model_training.py first.")
        return
    
    # Initialize fine-tuner
    fine_tuner = AdvancedFineTuner(
        base_model_path=base_model_path,
        output_dir="models/fine_tuned"
    )
    
    # Compare different fine-tuning approaches
    results = fine_tuner.compare_approaches(tokenized_dataset)
    
    # Save comparison results
    with open("models/fine_tuned/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Fine-tuning complete!")
    print("📁 Results saved to: models/fine_tuned/")

if __name__ == "__main__":
    main()
