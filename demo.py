"""
Quick Demo Script
A simple demonstration of the ML pipeline
"""

import sys
sys.path.append('src')

def quick_demo():
    """Run a quick demonstration of key concepts"""
    
    print("🎯 QUICK ML PIPELINE DEMO")
    print("=" * 50)
    
    # 1. Data Preparation Demo
    print("\n📊 1. DATA PREPARATION")
    print("-" * 30)
    
    from data_preparation import DataProcessor
    
    # Create synthetic data for quick demo
    processor = DataProcessor()
    
    # Sample texts for demo
    sample_data = [
        ("This movie is absolutely fantastic! Great acting and plot.", 1),
        ("Terrible film, complete waste of time and money.", 0),
        ("Amazing cinematography and outstanding performances.", 1),
        ("Boring and predictable storyline, very disappointing.", 0),
        ("One of the best movies I've ever seen! Highly recommend.", 1),
        ("Poor script and weak character development.", 0),
        ("Brilliant direction and compelling narrative structure.", 1),
        ("Not worth watching, very poor quality production.", 0),
        ("Excellent visual effects and great soundtrack too.", 1),
        ("Confusing plot and terrible acting throughout.", 0)
    ] * 50  # Repeat for more samples
    
    import pandas as pd
    df = pd.DataFrame(sample_data, columns=['text', 'label'])
    
    # Quick preprocessing
    processed_df = processor.preprocess_data(df)
    splits = processor.split_data(processed_df)
    
    print(f"✅ Created {len(processed_df)} samples")
    print(f"📊 Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
    
    # 2. Tokenization Demo
    print("\n🔤 2. TOKENIZATION")
    print("-" * 30)
    
    from tokenization import TextTokenizer
    
    # Simple tokenization example
    sample_text = "This movie was absolutely amazing!"
    
    print(f"Original text: {sample_text}")
    
    # Show tokenization without full model loading
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        tokens = tokenizer.tokenize(sample_text)
        token_ids = tokenizer.encode(sample_text)
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {tokenizer.decode(token_ids)}")
        
    except Exception as e:
        print(f"⚠️ Tokenizer demo skipped: {e}")
    
    # 3. Model Concepts Demo
    print("\n🤖 3. MODEL CONCEPTS")
    print("-" * 30)
    
    print("📚 Key Concepts:")
    print("  • BERT: Bidirectional transformer for understanding context")
    print("  • Fine-tuning: Adapting pre-trained models for specific tasks")
    print("  • LoRA: Low-Rank Adaptation - efficient parameter updates")
    print("  • Distillation: Creating smaller models from larger ones")
    
    # 4. Pipeline Steps Overview
    print("\n🔄 4. COMPLETE PIPELINE STEPS")
    print("-" * 30)
    
    steps = [
        ("Data Preparation", "Clean, preprocess, and split data"),
        ("Tokenization", "Convert text to model-readable format"),
        ("Model Training", "Train base model on your data"),
        ("Fine-tuning", "Efficient adaptation with LoRA"),
        ("Distillation", "Create smaller, faster models")
    ]
    
    for i, (step, description) in enumerate(steps, 1):
        print(f"  {i}. {step}: {description}")
    
    # 5. Benefits Summary
    print("\n🎯 5. BENEFITS OF THIS APPROACH")
    print("-" * 30)
    
    benefits = [
        "🚀 End-to-end pipeline from raw data to production model",
        "⚡ Efficient fine-tuning with LoRA (fewer parameters)",
        "📦 Model compression via knowledge distillation",
        "🔄 Reproducible and modular design",
        "📊 Comprehensive evaluation and comparison",
        "🛠️ Ready for production deployment"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    # 6. Next Steps
    print("\n📖 6. HOW TO USE THIS PROJECT")
    print("-" * 30)
    
    instructions = [
        "1. Install dependencies: pip install -r requirements.txt",
        "2. Run full pipeline: python run_pipeline.py",
        "3. Quick run: python run_pipeline.py --quick",
        "4. Explore interactively: Open notebooks/complete_pipeline.ipynb",
        "5. Run individual steps: python src/data_preparation.py",
        "6. Customize for your data and use case"
    ]
    
    for instruction in instructions:
        print(f"  {instruction}")
    
    print("\n" + "=" * 50)
    print("✅ DEMO COMPLETED!")
    print("🚀 Ready to run the full pipeline!")
    print("=" * 50)

if __name__ == "__main__":
    quick_demo()
