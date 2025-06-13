"""
Knowledge Distillation Module
This module implements knowledge distillation to create smaller, efficient models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path
import time

class KnowledgeDistiller:
    def __init__(self,
                 teacher_model_path: str,
                 student_model_name: str = "distilbert-base-uncased",
                 output_dir: str = "models/distilled",
                 temperature: float = 4.0,
                 alpha: float = 0.7):
        """
        Initialize Knowledge Distillation
        
        Args:
            teacher_model_path: Path to the teacher (large) model
            student_model_name: Name/path of the student (small) model
            output_dir: Directory to save distilled models
            temperature: Temperature for softmax distillation
            alpha: Weight for distillation loss vs hard target loss
        """
        self.teacher_model_path = teacher_model_path
        self.student_model_name = student_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft targets
        self.beta = 1.0 - alpha  # Weight for hard targets
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Load models
        self.load_teacher_model()
        self.load_student_model()
    
    def load_teacher_model(self):
        """Load the teacher model"""
        print(f"👨‍🏫 Loading teacher model from: {self.teacher_model_path}")
        
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path)
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            self.teacher_model_path
        ).to(self.device)
        
        # Set teacher to evaluation mode
        self.teacher_model.eval()
        
        # Freeze teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"✅ Teacher model loaded with {teacher_params:,} parameters")
    
    def load_student_model(self):
        """Load and initialize the student model"""
        print(f"👨‍🎓 Loading student model: {self.student_model_name}")
        
        # Load student tokenizer
        self.student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        # Load student model configuration
        config = AutoConfig.from_pretrained(
            self.student_model_name,
            num_labels=self.teacher_model.config.num_labels
        )
        
        # Load student model
        self.student_model = AutoModelForSequenceClassification.from_pretrained(
            self.student_model_name,
            config=config
        ).to(self.device)
        
        student_params = sum(p.numel() for p in self.student_model.parameters())
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        compression_ratio = student_params / teacher_params
        
        print(f"✅ Student model loaded with {student_params:,} parameters")
        print(f"📊 Compression ratio: {compression_ratio:.2%} of teacher size")
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Calculate distillation loss
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: True labels
        """
        # Soft target loss (knowledge distillation)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + self.beta * hard_loss
        
        return total_loss, distillation_loss, hard_loss
    
    class DistillationTrainer(Trainer):
        """Custom trainer for knowledge distillation"""
        
        def __init__(self, teacher_model, temperature, alpha, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model
            self.temperature = temperature
            self.alpha = alpha
            self.beta = 1.0 - alpha
        
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Compute distillation loss
            """
            labels = inputs.pop("labels")
            
            # Student forward pass
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Calculate distillation loss
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            
            distillation_loss = F.kl_div(
                soft_student, 
                soft_teacher, 
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Hard target loss
            hard_loss = F.cross_entropy(student_logits, labels)
            
            # Combined loss
            loss = self.alpha * distillation_loss + self.beta * hard_loss
            
            return (loss, student_outputs) if return_outputs else loss
    
    def distill_model(self,
                     tokenized_dataset,
                     num_epochs: int = 5,
                     batch_size: int = 16,
                     learning_rate: float = 5e-5):
        """
        Perform knowledge distillation
        
        Args:
            tokenized_dataset: Tokenized dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print("🧠 Starting knowledge distillation...")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,
            dataloader_pin_memory=False,
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.student_tokenizer,
            padding=True
        )
        
        # Create custom trainer
        trainer = self.DistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=self.temperature,
            alpha=self.alpha,
            model=self.student_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.student_tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("🏃‍♂️ Training student model...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Evaluate
        eval_result = trainer.evaluate(tokenized_dataset["test"])
        
        # Save model
        trainer.save_model()
        self.student_tokenizer.save_pretrained(str(self.output_dir))
        
        print(f"⏱️ Distillation completed in {training_time:.2f} seconds")
        print("📊 Distillation Results:")
        print(f"  - Train Loss: {train_result.training_loss:.4f}")
        print(f"  - Test Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"  - Test F1: {eval_result['eval_f1']:.4f}")
        
        # Save results
        results = {
            "train_loss": train_result.training_loss,
            "test_results": eval_result,
            "training_time": training_time,
            "teacher_model": self.teacher_model_path,
            "student_model": self.student_model_name,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "compression_ratio": sum(p.numel() for p in self.student_model.parameters()) / 
                               sum(p.numel() for p in self.teacher_model.parameters())
        }
        
        with open(self.output_dir / "distillation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return trainer, results
    
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
    
    def compare_models(self, tokenized_dataset):
        """Compare teacher and student models"""
        print("\n🔍 Comparing Teacher vs Student Performance...")
        
        # Evaluate teacher
        teacher_results = self.evaluate_model(
            self.teacher_model, 
            self.teacher_tokenizer,
            tokenized_dataset["test"],
            "Teacher"
        )
        
        # Evaluate student
        student_results = self.evaluate_model(
            self.student_model,
            self.student_tokenizer, 
            tokenized_dataset["test"],
            "Student"
        )
        
        # Print comparison
        self.print_model_comparison(teacher_results, student_results)
        
        return teacher_results, student_results
    
    def evaluate_model(self, model, tokenizer, test_dataset, model_name):
        """Evaluate a single model"""
        model.eval()
        
        predictions = []
        labels = []
        
        # Create data loader
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        dataloader = DataLoader(
            test_dataset, 
            batch_size=32, 
            collate_fn=data_collator
        )
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                batch_labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get predictions
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
        
        return results
    
    def print_model_comparison(self, teacher_results, student_results):
        """Print comparison between teacher and student"""
        print("\n" + "="*60)
        print("📊 TEACHER vs STUDENT COMPARISON")
        print("="*60)
        
        print(f"{'Metric':<15} {'Teacher':<15} {'Student':<15} {'Retention':<15}")
        print("-" * 60)
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        for metric in metrics:
            teacher_val = teacher_results[metric]
            student_val = student_results[metric]
            retention = (student_val / teacher_val) * 100 if teacher_val > 0 else 0
            
            print(f"{metric.title():<15} "
                  f"{teacher_val:<15.4f} "
                  f"{student_val:<15.4f} "
                  f"{retention:<15.1f}%")
        
        # Model size comparison
        teacher_params = teacher_results['num_parameters']
        student_params = student_results['num_parameters']
        compression = (student_params / teacher_params) * 100
        
        print(f"\n📏 Model Size:")
        print(f"Teacher: {teacher_params:,} parameters")
        print(f"Student: {student_params:,} parameters")
        print(f"Compression: {compression:.1f}% of original size")
        print(f"Size reduction: {100-compression:.1f}%")
    
    def test_inference_speed(self, test_texts: list):
        """Compare inference speed between teacher and student"""
        print("\n⚡ Comparing Inference Speed...")
        
        def measure_inference_time(model, tokenizer, texts, num_runs=10):
            model.eval()
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                
                for text in texts:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            return np.mean(times), np.std(times)
        
        # Measure teacher speed
        teacher_mean, teacher_std = measure_inference_time(
            self.teacher_model, self.teacher_tokenizer, test_texts
        )
        
        # Measure student speed
        student_mean, student_std = measure_inference_time(
            self.student_model, self.student_tokenizer, test_texts
        )
        
        speedup = teacher_mean / student_mean
        
        print(f"Teacher inference time: {teacher_mean:.4f} ± {teacher_std:.4f} seconds")
        print(f"Student inference time: {student_mean:.4f} ± {student_std:.4f} seconds")
        print(f"Speedup: {speedup:.2f}x faster")
        
        return speedup

def main():
    """Main function to run knowledge distillation"""
    # Load tokenized dataset
    try:
        tokenized_dataset = load_from_disk("data/processed/tokenized_dataset")
        print("✅ Loaded tokenized dataset")
    except:
        print("❌ No tokenized dataset found. Please run previous steps first.")
        return
    
    # Check if teacher model exists
    teacher_model_path = "models/trained/bert_classifier"
    if not Path(teacher_model_path).exists():
        print("❌ No teacher model found. Please run model_training.py first.")
        return
    
    # Initialize distiller
    distiller = KnowledgeDistiller(
        teacher_model_path=teacher_model_path,
        student_model_name="distilbert-base-uncased",
        output_dir="models/distilled/distilbert_student",
        temperature=4.0,
        alpha=0.7
    )
    
    # Perform distillation
    trainer, results = distiller.distill_model(
        tokenized_dataset,
        num_epochs=3,
        batch_size=16,
        learning_rate=5e-5
    )
    
    # Compare models
    teacher_results, student_results = distiller.compare_models(tokenized_dataset)
    
    # Test inference speed
    test_texts = [
        "This movie was absolutely fantastic!",
        "I didn't like this film at all.",
        "It was an okay movie, nothing special."
    ]
    
    speedup = distiller.test_inference_speed(test_texts)
    
    print("\n✅ Knowledge distillation complete!")
    print(f"📁 Distilled model saved to: {distiller.output_dir}")
    print(f"⚡ Student model is {speedup:.2f}x faster than teacher")

if __name__ == "__main__":
    main()
