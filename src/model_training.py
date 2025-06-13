"""
Model Training Module
This module handles the base model training process.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time

class ModelTrainer:
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 num_labels: int = 2,
                 output_dir: str = "models/trained"):
        """
        Initialize model trainer
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of classification labels
            output_dir: Directory to save trained models
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.load_model()
        self.load_tokenizer()
    
    def load_model(self):
        """Load pre-trained model"""
        print(f"🤖 Loading model: {self.model_name}")
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            finetuning_task="text-classification"
        )
        
        # Load model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=self.config
        )
        
        # Move to device
        self.model.to(self.device)
        
        print(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
    
    def load_tokenizer(self):
        """Load tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def create_training_arguments(self, 
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 16,
                                per_device_eval_batch_size: int = 64,
                                learning_rate: float = 2e-5):        """Create training arguments"""
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",  # Updated from evaluation_strategy
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb for demo
            use_cpu=not torch.cuda.is_available(),  # Updated from fp16
        )
        
        return training_args
    
    def train_model(self, 
                   tokenized_dataset,
                   num_epochs: int = 3,
                   batch_size: int = 16,
                   learning_rate: float = 2e-5):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Create training arguments
        training_args = self.create_training_arguments(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Training
        print("🏃‍♂️ Training in progress...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Save model
        trainer.save_model()
        
        # Evaluate on test set
        test_results = trainer.evaluate(tokenized_dataset["test"])
        
        # Save training results
        results = {
            "train_loss": train_result.training_loss,
            "test_results": test_results,
            "training_time": training_time,
            "model_name": self.model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        
        with open(self.output_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("📊 Training Results:")
        print(f"  - Train Loss: {train_result.training_loss:.4f}")
        print(f"  - Test Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"  - Test F1: {test_results['eval_f1']:.4f}")
        
        return trainer, results
    
    def evaluate_model(self, trainer, tokenized_dataset):
        """Detailed model evaluation"""
        print("\n📈 Detailed Model Evaluation...")
        
        # Get predictions
        predictions = trainer.predict(tokenized_dataset["test"])
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        print(f"\n📊 Detailed Metrics:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        for i in range(self.num_labels):
            print(f"Class {i}:")
            print(f"  - Precision: {precision[i]:.4f}")
            print(f"  - Recall: {recall[i]:.4f}")
            print(f"  - F1: {f1[i]:.4f}")
            print(f"  - Support: {support[i]}")
        
        return {
            "accuracy": accuracy,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist()
        }
    
    def test_inference(self, texts: list):
        """Test model inference on new texts"""
        print("\n🔍 Testing Model Inference...")
        
        self.model.eval()
        results = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            results.append({
                "text": text,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": predictions[0].tolist()
            })
            
            print(f"Text: {text[:100]}...")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print("-" * 50)
        
        return results
    
    def plot_training_history(self, trainer):
        """Plot training history"""
        try:
            logs = trainer.state.log_history
            
            train_losses = []
            eval_losses = []
            eval_accuracies = []
            steps = []
            
            for log in logs:
                if 'loss' in log:
                    train_losses.append(log['loss'])
                    steps.append(log['step'])
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
                if 'eval_accuracy' in log:
                    eval_accuracies.append(log['eval_accuracy'])
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot losses
            ax1.plot(steps, train_losses, label='Training Loss', color='blue')
            if eval_losses:
                eval_steps = steps[-len(eval_losses):]
                ax1.plot(eval_steps, eval_losses, label='Validation Loss', color='red')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            if eval_accuracies:
                eval_steps = steps[-len(eval_accuracies):]
                ax2.plot(eval_steps, eval_accuracies, label='Validation Accuracy', color='green')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Validation Accuracy')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "training_history.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Training plots saved to {self.output_dir / 'training_history.png'}")
            
        except Exception as e:
            print(f"❌ Could not plot training history: {e}")

def main():
    """Main function to run model training"""
    # Load tokenized dataset
    try:
        tokenized_dataset = load_from_disk("data/processed/tokenized_dataset")
        print("✅ Loaded tokenized dataset")
    except:
        print("❌ No tokenized dataset found. Please run tokenization.py first.")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name="bert-base-uncased",
        num_labels=2,
        output_dir="models/trained/bert_classifier"
    )
    
    # Train model
    trained_model, results = trainer.train_model(
        tokenized_dataset,
        num_epochs=2,  # Small number for demo
        batch_size=8,  # Small batch size for demo
        learning_rate=2e-5
    )
    
    # Detailed evaluation
    eval_results = trainer.evaluate_model(trained_model, tokenized_dataset)
    
    # Test inference
    test_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film, waste of time and money.",
        "It was okay, nothing special but not bad either."
    ]
    
    inference_results = trainer.test_inference(test_texts)
    
    # Plot training history
    trainer.plot_training_history(trained_model)
    
    print("\n✅ Model training complete!")
    print(f"📁 Model saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()
