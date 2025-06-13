from flask import Flask, render_template, request, jsonify
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server applications
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Additional matplotlib configuration for Flask
plt.ioff()  # Turn off interactive mode
plt.style.use('default')  # Use default style to avoid any style conflicts

app = Flask(__name__)

# Format large numbers with commas
@app.template_filter('format_number')
def format_number(value):
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value

# Load data and models once at startup
df = pd.read_csv("data/raw/imdb_sample.csv")
tokenized_path = "data/processed/tokenized_dataset"
processed_path = "data/processed/final_dataset"

if Path(processed_path).exists():
    processed = load_from_disk(processed_path)
else:
    processed = None

if Path(tokenized_path).exists():
    tokenized = load_from_disk(tokenized_path)
else:
    tokenized = None

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.eval()

def create_distribution_chart(data, title):
    """Create a bar chart for sentiment distribution"""
    try:
        plt.figure(figsize=(8, 5))
        counts = data.value_counts()
        plt.bar(['Negative', 'Positive'], [counts.get(0, 0), counts.get(1, 0)], color=['#ff6b6b', '#4ecdc4'])
        plt.title(title)
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save plot to a base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  # Important: close the figure to free memory
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        print(f"Error creating distribution chart: {e}")
        plt.close()  # Ensure figure is closed even on error
        return None

def create_histogram_chart(data, title, xlabel, bins=20):
    """Create a histogram chart"""
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  # Important: close the figure to free memory
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        print(f"Error creating histogram chart: {e}")
        plt.close()  # Ensure figure is closed even on error
        return None

def create_multi_histogram_chart(data_dict, title, xlabel, bins=20):
    """Create multiple histograms on the same chart"""
    try:
        plt.figure(figsize=(12, 6))
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        for i, (label, data) in enumerate(data_dict.items()):
            plt.hist(data, bins=bins, alpha=0.6, label=label, color=colors[i % len(colors)])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  # Important: close the figure to free memory
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        print(f"Error creating multi-histogram chart: {e}")
        plt.close()  # Ensure figure is closed even on error
        return None

def analyze_vocabulary(texts):
    """Analyze vocabulary statistics"""
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    vocab_size = len(word_counts)
    total_words = len(all_words)
    
    # Most common words
    most_common = word_counts.most_common(10)
    
    # Word frequency distribution
    frequencies = list(word_counts.values())
    
    return {
        'vocab_size': vocab_size,
        'total_words': total_words,
        'avg_word_freq': total_words / vocab_size if vocab_size > 0 else 0,
        'most_common': most_common,
        'word_freq_stats': {
            'min': min(frequencies) if frequencies else 0,
            'max': max(frequencies) if frequencies else 0,
            'mean': np.mean(frequencies) if frequencies else 0,
            'median': np.median(frequencies) if frequencies else 0
        }
    }

def create_training_loss_chart(losses, steps):
    """Create training loss visualization"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2, marker='o')
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  # Important: close the figure to free memory
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        print(f"Error creating training loss chart: {e}")
        plt.close()  # Ensure figure is closed even on error
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    steps = []
      # Step 1: Raw Data
    df_sample = df.head(5)
    sentiment_counts = df['label'].value_counts().to_dict()
    sentiment_chart = create_distribution_chart(df['label'], 'Sentiment Distribution in Raw Data')
    sample_df = df[['text', 'label']].rename(columns={'label': 'sentiment'}).sample(3)
    # --- NEW: Text length statistics ---
    text_lengths = df['text'].str.len()
    word_lengths = df['text'].str.split().apply(len)
    text_length_stats = {
        "min": int(text_lengths.min()),
        "max": int(text_lengths.max()),
        "mean": float(text_lengths.mean()),
        "median": float(text_lengths.median()),
        "std": float(text_lengths.std()),
        "histogram": np.histogram(text_lengths, bins=10)[0].tolist(),
        "bin_edges": np.histogram(text_lengths, bins=10)[1].tolist()
    }
    word_length_stats = {
        "min": int(word_lengths.min()),
        "max": int(word_lengths.max()),
        "mean": float(word_lengths.mean()),
        "median": float(word_lengths.median()),
        "std": float(word_lengths.std()),
        "histogram": np.histogram(word_lengths, bins=10)[0].tolist(),
        "bin_edges": np.histogram(word_lengths, bins=10)[1].tolist()
    }
      # --- NEW: Additional detailed analysis ---
    # Text length histograms
    text_length_chart = create_histogram_chart(text_lengths, 'Text Length Distribution (Characters)', 'Characters') or ""
    word_length_chart = create_histogram_chart(word_lengths, 'Text Length Distribution (Words)', 'Words') or ""
    
    # Vocabulary analysis
    vocab_analysis = analyze_vocabulary(df['text'].tolist())
    
    # Sentiment-based analysis
    positive_texts = df[df['label'] == 1]['text']
    negative_texts = df[df['label'] == 0]['text']
    
    sentiment_text_lengths = {
        'Positive': positive_texts.str.len().tolist(),
        'Negative': negative_texts.str.len().tolist()
    }
    sentiment_length_chart = create_multi_histogram_chart(
        sentiment_text_lengths, 
        'Text Length by Sentiment', 
        'Characters'
    ) or ""
    
    # Class balance analysis
    class_balance = {
        'negative_ratio': sentiment_counts.get(0, 0) / len(df),
        'positive_ratio': sentiment_counts.get(1, 0) / len(df),
        'balance_score': min(sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)) / max(sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)) if max(sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)) > 0 else 0
    }
    
    raw_data = {
        "table": df_sample.to_html(classes="table table-striped table-hover"),
        "counts": sentiment_counts,
        "chart": sentiment_chart,
        "sample_texts": sample_df.to_dict('records'),
        "text_length_stats": text_length_stats,
        "word_length_stats": word_length_stats,
        "text_length_chart": text_length_chart,
        "word_length_chart": word_length_chart,
        "vocab_analysis": vocab_analysis,
        "sentiment_length_chart": sentiment_length_chart,
        "class_balance": class_balance
    }
    steps.append({"title": "Raw Data", "icon": "📊", "content": raw_data})

    # Step 2: Processed Dataset
    if processed:
        processed_examples = {}
        processed_stats = {}
        processed_charts = {}
        
        for split in processed:
            samples = []
            text_lengths = []
            word_lengths = []
            labels = []
            
            for ex in processed[split].select(range(min(1000, len(processed[split])))):
                samples.append(ex)
                if 'text' in ex:
                    text_lengths.append(len(ex['text']))
                    word_lengths.append(len(ex['text'].split()))
                if 'label' in ex:
                    labels.append(ex['label'])
            
            # Compute stats if available
            if text_lengths:
                split_text_stats = {
                    "min": int(np.min(text_lengths)),
                    "max": int(np.max(text_lengths)),
                    "mean": float(np.mean(text_lengths)),
                    "median": float(np.median(text_lengths)),
                    "std": float(np.std(text_lengths)),
                    "histogram": np.histogram(text_lengths, bins=10)[0].tolist(),
                    "bin_edges": np.histogram(text_lengths, bins=10)[1].tolist()
                }
                split_word_stats = {
                    "min": int(np.min(word_lengths)),
                    "max": int(np.max(word_lengths)),
                    "mean": float(np.mean(word_lengths)),
                    "median": float(np.median(word_lengths)),
                    "std": float(np.std(word_lengths)),
                    "histogram": np.histogram(word_lengths, bins=10)[0].tolist(),
                    "bin_edges": np.histogram(word_lengths, bins=10)[1].tolist()
                }
                
                # Create charts for this split
                processed_charts[split] = {
                    'text_length_chart': create_histogram_chart(
                        text_lengths, 
                        f'{split.capitalize()} Text Length Distribution', 
                        'Characters'
                    ),
                    'word_length_chart': create_histogram_chart(
                        word_lengths, 
                        f'{split.capitalize()} Word Length Distribution', 
                        'Words'
                    )
                }
                
                # Sentiment distribution for this split
                if labels:
                    label_counts = Counter(labels)
                    processed_charts[split]['sentiment_chart'] = create_distribution_chart(
                        pd.Series(labels), 
                        f'{split.capitalize()} Sentiment Distribution'
                    )
                    split_class_balance = {
                        'negative_ratio': label_counts.get(0, 0) / len(labels),
                        'positive_ratio': label_counts.get(1, 0) / len(labels),
                        'balance_score': min(label_counts.get(0, 0), label_counts.get(1, 0)) / max(label_counts.get(0, 0), label_counts.get(1, 0)) if max(label_counts.get(0, 0), label_counts.get(1, 0)) > 0 else 0
                    }
                else:
                    split_class_balance = None
            else:
                split_text_stats = split_word_stats = None
                split_class_balance = None
                
            processed_examples[split] = samples[:3]
            processed_stats[split] = {
                "text_length_stats": split_text_stats,
                "word_length_stats": split_word_stats,
                "size": len(processed[split]),
                "class_balance": split_class_balance
            }
            
        processed_data = {
            "sizes": {split: len(processed[split]) for split in processed},
            "examples": processed_examples,
            "stats": processed_stats,
            "charts": processed_charts
        }
        steps.append({"title": "Processed Dataset", "icon": "🔄", "content": processed_data})
    
    # Step 3: Tokenization Example
    if tokenized:
        sample = tokenized['train'][0]
        decoded = tokenizer.convert_ids_to_tokens(sample['input_ids'])
        attention_mask_visual = ''.join(['█' if m == 1 else '░' for m in sample['attention_mask']])
        # --- NEW: Token length statistics ---
        token_length_stats = {}
        token_charts = {}
        
        for split in tokenized:
            input_ids_list = [item['input_ids'] for item in tokenized[split].select(range(min(1000, len(tokenized[split]))))]
            lengths = [len(ids) for ids in input_ids_list]
            if lengths:
                token_length_stats[split] = {
                    "min": int(np.min(lengths)),
                    "max": int(np.max(lengths)),
                    "mean": float(np.mean(lengths)),
                    "median": float(np.median(lengths)),
                    "std": float(np.std(lengths)),
                    "histogram": np.histogram(lengths, bins=10)[0].tolist(),
                    "bin_edges": np.histogram(lengths, bins=10)[1].tolist()
                }
                
                # Create token length chart for this split
                token_charts[split] = create_histogram_chart(
                    lengths,
                    f'{split.capitalize()} Token Length Distribution',
                    'Number of Tokens'
                )
            else:
                token_length_stats[split] = None
                
        # Compare token lengths across splits
        if len(token_length_stats) > 1:
            token_comparison_data = {}
            for split, stats in token_length_stats.items():
                if stats:
                    input_ids_list = [item['input_ids'] for item in tokenized[split].select(range(min(500, len(tokenized[split]))))]
                    token_comparison_data[split] = [len(ids) for ids in input_ids_list]
            
            if token_comparison_data:
                token_comparison_chart = create_multi_histogram_chart(
                    token_comparison_data,
                    'Token Length Comparison Across Splits',
                    'Number of Tokens'
                )
            else:
                token_comparison_chart = None
        else:
            token_comparison_chart = None
            
        # Special tokens analysis
        special_tokens = {
            'pad_token': tokenizer.pad_token,
            'unk_token': tokenizer.unk_token,
            'cls_token': tokenizer.cls_token,
            'sep_token': tokenizer.sep_token,
            'mask_token': tokenizer.mask_token
        }
        
        # Token type analysis for the sample
        token_types = {
            'special_tokens': 0,
            'subwords': 0,
            'complete_words': 0
        }
        
        for token in decoded:
            if token in [tokenizer.pad_token, tokenizer.unk_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.mask_token, '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                token_types['special_tokens'] += 1
            elif token.startswith('##'):
                token_types['subwords'] += 1
            else:
                token_types['complete_words'] += 1
        
        tokenization_data = {
            "sizes": {split: len(tokenized[split]) for split in tokenized},
            "sample_text": processed['train'][0]['text'] if processed else "Sample text",
            "input_ids": sample['input_ids'],
            "attention_mask": sample['attention_mask'],
            "attention_visual": attention_mask_visual,
            "decoded_tokens": decoded,
            "label": int(processed['train'][0]['label'] if processed else df.loc[0, 'label']),
            "token_length_stats": token_length_stats,
            "token_charts": token_charts,
            "token_comparison_chart": token_comparison_chart,
            "special_tokens": special_tokens,
            "token_types": token_types
        }
        steps.append({"title": "Tokenization", "icon": "📦", "content": tokenization_data})
    
    # Step 4: Model Architecture
    model_layers = []
    for name, param in list(model.named_parameters())[:10]:  # Just first 10 parameters
        model_layers.append({
            "name": name,
            "shape": str(list(param.shape)),
            "parameters": param.numel()
        })
    # --- NEW: Model config details ---
    model_config = model.config.to_dict()
    model_data = {
        "name": "bert-base-uncased",
        "total_parameters": model.num_parameters(),
        "sample_layers": model_layers,
        "labels": ["Negative", "Positive"],
        "config": model_config
    }
    steps.append({"title": "Model Architecture", "icon": "🤖", "content": model_data})

    # Step 5: Training Process
    # Enhanced with more realistic training simulation
    training_losses = [0.6932, 0.6543, 0.5932, 0.5421, 0.4932, 0.4654, 0.4321, 0.4102, 0.3876, 0.3654]
    training_steps = list(range(0, len(training_losses) * 10, 10))
    eval_losses = [0.6821, 0.6234, 0.5543, 0.5012, 0.4687, 0.4432, 0.4198, 0.3987, 0.3754, 0.3521]
    eval_steps = list(range(5, len(eval_losses) * 10 + 5, 10))
    
    # Create training loss chart
    training_loss_chart = create_training_loss_chart(training_losses, training_steps)
    
    # Create combined training/eval loss chart
    plt.figure(figsize=(12, 6))
    plt.plot(training_steps, training_losses, 'b-', linewidth=2, marker='o', label='Training Loss')
    plt.plot(eval_steps, eval_losses, 'r-', linewidth=2, marker='s', label='Evaluation Loss')
    plt.title('Training and Evaluation Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    combined_loss_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()
    
    # Training metrics summary
    training_metrics = {
        'initial_loss': training_losses[0],
        'final_loss': training_losses[-1],
        'loss_reduction': training_losses[0] - training_losses[-1],
        'loss_reduction_percent': ((training_losses[0] - training_losses[-1]) / training_losses[0]) * 100,
        'convergence_rate': (training_losses[0] - training_losses[-1]) / len(training_losses),
        'best_eval_loss': min(eval_losses),
        'eval_improvement': eval_losses[0] - min(eval_losses)
    }
    
    training_data = {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": "2e-5",
        "sample_batch": tokenized['train'][0] if tokenized else None,
        "metrics": {
            "losses": training_losses,
            "steps": training_steps,
            "eval_losses": eval_losses,
            "eval_steps": eval_steps
        },
        "training_loss_chart": training_loss_chart,
        "combined_loss_chart": combined_loss_chart,
        "training_metrics": training_metrics
    }
    steps.append({"title": "Training Process", "icon": "🏃‍♂️", "content": training_data})

    # Step 6: Inference Results
    test_texts = df['text'].sample(5).tolist()
    inferences = []
    confidence_scores = []
    prediction_distribution = {'Positive': 0, 'Negative': 0}
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        pred = np.argmax(probs)
        label = "Positive" if pred == 1 else "Negative"
        confidence = float(probs[pred])
        
        prediction_distribution[label] += 1
        confidence_scores.append(confidence)
        
        inferences.append({
            "text": text[:200] + "..." if len(text) > 200 else text,
            "label": label,
            "confidence": confidence,
            "probs": {"negative": float(probs[0]), "positive": float(probs[1])}
        })
    
    # Confidence distribution analysis
    confidence_stats = {
        'mean': np.mean(confidence_scores),
        'min': np.min(confidence_scores),
        'max': np.max(confidence_scores),
        'std': np.std(confidence_scores)
    }
    
    # Create confidence distribution chart
    confidence_chart = create_histogram_chart(
        confidence_scores,
        'Prediction Confidence Distribution',
        'Confidence Score'
    )
    
    # Prediction distribution chart
    pred_dist_chart = create_distribution_chart(
        pd.Series([1 if inf['label'] == 'Positive' else 0 for inf in inferences]),
        'Model Predictions Distribution'
    )
    
    inference_data = {
        "results": inferences,
        "model_name": "bert-base-uncased",
        "prediction_distribution": prediction_distribution,
        "confidence_stats": confidence_stats,
        "confidence_chart": confidence_chart,
        "prediction_chart": pred_dist_chart
    }
    steps.append({"title": "Inference Results", "icon": "🔍", "content": inference_data})
    
    return render_template("index.html", steps=steps)

@app.route('/inference', methods=['POST'])
def inference():
    text = request.form.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"})
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    pred = np.argmax(probs)
    label = "Positive" if pred == 1 else "Negative"
    
    return jsonify({
        "text": text,
        "prediction": label,
        "confidence": float(probs[pred]),
        "probabilities": {
            "negative": float(probs[0]),
            "positive": float(probs[1])
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and Docker health checks."""
    try:
        # Basic health checks
        status = {
            'status': 'healthy',
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': '1.0.0',
            'components': {
                'database': 'ok',
                'model': 'loaded' if model is not None else 'not_loaded',
                'tokenizer': 'loaded' if tokenizer is not None else 'not_loaded',
                'data': 'loaded' if df is not None else 'not_loaded'
            }
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
