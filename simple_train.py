"""
Simple Model Training Script
A simplified version for demonstration
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

def compute_metrics(eval_pred):
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

def main():
    """Simple training function"""
    print("🤖 Starting simple model training...")
      # Load tokenized dataset
    try:
        tokenized_dataset = load_from_disk("data/processed/tokenized_dataset")
        print("✅ Loaded tokenized dataset")
    except Exception as e:
        print(f"❌ No tokenized dataset found: {e}. Please run tokenization first.")
        return
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    print(f"🤖 Model loaded with {model.num_parameters():,} parameters")
      # Create training arguments with updated API
    output_dir = Path("models/trained/simple_bert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,  # Quick training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",  # Updated API
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=[],  # Explicitly disable all reporting
        use_cpu=not torch.cuda.is_available(),
    )
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
      # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,  # Updated API
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("🏃‍♂️ Starting training...")
    train_result = trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate(tokenized_dataset["test"])
    
    # Save model
    trainer.save_model()
    
    print("📊 Training Results:")
    print(f"  - Train Loss: {train_result.training_loss:.4f}")
    print(f"  - Test Accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"  - Test F1: {eval_result['eval_f1']:.4f}")
    
    # Test inference
    test_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film, complete waste of time and money.",
        "It was okay, nothing special but not bad either."
    ]
    
    print("\n🔍 Testing Model Inference...")
    model.eval()
    
    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        print("-" * 50)
    
    print(f"\n✅ Training complete! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
