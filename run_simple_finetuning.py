import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import numpy as np
from pathlib import Path
import json # For structured output

def get_text_length_stats(texts):
    lengths = [len(text) for text in texts]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0, "histogram": {}}
    
    hist, bin_edges = np.histogram(lengths, bins=10)
    histogram_data = {f"{int(bin_edges[i])}-{int(bin_edges[i+1])}": int(count) for i, count in enumerate(hist)}
    
    return {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "histogram": histogram_data
    }

def get_token_length_stats(input_ids_list):
    lengths = [len(ids) for ids in input_ids_list]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0, "histogram": {}}

    hist, bin_edges = np.histogram(lengths, bins=10)
    histogram_data = {f"{int(bin_edges[i])}-{int(bin_edges[i+1])}": int(count) for i, count in enumerate(hist)}

    return {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "histogram": histogram_data
    }

def run_pipeline():
    pipeline_data = {}

    print("🔍 Step 1: Load Raw Data")
    df = pd.read_csv("data/raw/imdb_sample.csv")
    raw_data_stats = {
        "head": df.head().to_dict(),
        "count": len(df),
        "text_length_stats": get_text_length_stats(df['text'].tolist())
    }
    pipeline_data["raw_data"] = raw_data_stats
    print(json.dumps({"raw_data_stats": raw_data_stats}, indent=2))


    print("\n🔄 Step 2: Load Processed Dataset")
    processed_path = "data/processed/final_dataset"
    processed_data_details = {"path": processed_path, "exists": False, "splits": {}}
    if Path(processed_path).exists():
        processed_data_details["exists"] = True
        processed = load_from_disk(processed_path)
        for split_name, split_data in processed.items():
            examples = [ex for ex in split_data.select(range(min(3, len(split_data))))]
            # Assuming 'text' column exists after processing
            text_column = 'text' if 'text' in split_data.column_names else (split_data.column_names[0] if split_data.column_names else None)
            
            split_stats = {
                "size": len(split_data),
                "examples": examples,
            }
            if text_column:
                 split_stats["text_length_stats"] = get_text_length_stats(split_data[text_column])
            else:
                split_stats["text_length_stats"] = "N/A (No 'text' column found)"

            processed_data_details["splits"][split_name] = split_stats
    else:
        print(f"No processed dataset found at {processed_path}")
    pipeline_data["processed_data"] = processed_data_details
    print(json.dumps({"processed_data_details": processed_data_details}, indent=2))


    print("\n📦 Step 3: Load Tokenized Dataset")
    tokenized_path = "data/processed/tokenized_dataset"
    tokenized_data_details = {"path": tokenized_path, "exists": False, "splits": {}}
    if Path(tokenized_path).exists():
        tokenized_data_details["exists"] = True
        tokenized = load_from_disk(tokenized_path)
        tmp_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # For decoding examples
        for split_name, split_data in tokenized.items():
            sample_input_ids = split_data[0]['input_ids'] if len(split_data) > 0 else []
            decoded_sample = tmp_tokenizer.convert_ids_to_tokens(sample_input_ids) if sample_input_ids else []
            
            # Get token length stats for a sample of the data to avoid long processing
            sample_size_for_stats = min(1000, len(split_data))
            sampled_input_ids = [item['input_ids'] for item in split_data.select(range(sample_size_for_stats))]

            tokenized_data_details["splits"][split_name] = {
                "size": len(split_data),
                "example_input_ids": sample_input_ids,
                "example_decoded_tokens": decoded_sample,
                "token_length_stats": get_token_length_stats(sampled_input_ids)
            }
    else:
        print(f"No tokenized dataset found at {tokenized_path}")
    pipeline_data["tokenized_data"] = tokenized_data_details
    print(json.dumps({"tokenized_data_details": tokenized_data_details}, indent=2))

    print("\n🔧 Step 4: Initialize Model and Tokenizer")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_details = {
        "name": model_name,
        "num_parameters": model.num_parameters(),
        "config": model.config.to_dict() # Includes hidden_size, num_attention_heads, num_hidden_layers etc.
    }
    pipeline_data["model_details"] = model_details
    print(json.dumps({"model_details": model_details}, indent=2))


    print("\n🏃‍♂️ Step 5: Training Preview")
    training_preview_details = {}
    training_args = TrainingArguments(
        output_dir="models/trained/simple_bert_preview", # Potentially make this unique per run or configurable
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        logging_steps=10,
        # eval_strategy="steps", # Ensure eval_steps is used with this
        eval_steps=10, # Evaluate every 10 steps
        save_strategy="no", # No need to save preview model checkpoints
        report_to=[], # Disable external reporting like wandb for preview
        use_cpu=not torch.cuda.is_available(),
        # Ensure tokenized dataset is loaded for this step
        # load_best_model_at_end=True, # Optional: if you want to load the best model based on eval
        # metric_for_best_model="loss", # Optional: if load_best_model_at_end is True
    )
    training_preview_details["training_args"] = training_args.to_dict()
    
    if tokenized_data_details["exists"] and 'train' in tokenized and 'validation' in tokenized:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        small_train_dataset = tokenized['train'].select(range(min(20, len(tokenized['train']))))
        small_eval_dataset = tokenized['validation'].select(range(min(20, len(tokenized['validation']))))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics, # Add if you have a metrics function
        )
        train_result = trainer.train()
        training_preview_details["training_loss"] = train_result.training_loss
        training_preview_details["log_history"] = trainer.state.log_history
        print(f"Preview training completed. Loss: {train_result.training_loss:.4f}")
        print(f"Log history: {trainer.state.log_history}")

    else:
        training_preview_details["status"] = "Skipped: Tokenized data not available."
        print("Skipping training preview as tokenized data is not available.")
    
    pipeline_data["training_preview"] = training_preview_details
    print(json.dumps({"training_preview_details": training_preview_details}, indent=2, default=str))


    print("\n🔍 Step 6: Inference Preview")
    inference_preview_details = {"samples": []}
    # Ensure df is available from Step 1
    if 'text' in df.columns:
        test_texts = df['text'].sample(min(3, len(df))).tolist()
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            # Move inputs to the same device as the model
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad(): # Ensure no gradients are computed during inference
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_label_idx = np.argmax(probs)
            # Assuming binary classification: 0 for Negative, 1 for Positive
            label_map = {0: "Negative", 1: "Positive"} 
            predicted_label = label_map.get(pred_label_idx, f"Unknown_Label_{pred_label_idx}")
            
            inference_preview_details["samples"].append({
                "text": text,
                "predicted_label": predicted_label,
                "confidence": float(probs[pred_label_idx]),
                "probabilities": {"Negative": float(probs[0]), "Positive": float(probs[1])}
            })
        print(f"Inference samples: {json.dumps(inference_preview_details['samples'], indent=2)}")
    else:
        inference_preview_details["status"] = "Skipped: Raw data with 'text' column not available."
        print("Skipping inference preview as raw data is not available.")
        
    pipeline_data["inference_preview"] = inference_preview_details
    # print(json.dumps({"inference_preview_details": inference_preview_details}, indent=2)) # Already printed above

    # At the end, print all collected data as a single JSON object
    # This can be captured by app.py
    print("\n--- PIPELINE DATA ---")
    # Need a custom serializer for things like PosixPath or other non-serializable objects if they sneak in
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, 'to_dict'): # For HuggingFace TrainingArguments etc.
                return obj.to_dict()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    print(json.dumps(pipeline_data, indent=2, cls=CustomEncoder))
    return pipeline_data


if __name__ == "__main__":
    run_pipeline()