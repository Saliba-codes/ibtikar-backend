import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import unicodedata

# ============================================================================
# PREPROCESSING (Same as training)
# ============================================================================

def preprocess_text(text):
    """
    Apply the same preprocessing used during training
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    txt = text
    
    # Unicode normalization
    txt = unicodedata.normalize("NFKC", txt)
    
    # Remove URLs
    txt = re.sub(r'http\S+|www\.\S+', ' ', txt)
    
    # Remove email addresses
    txt = re.sub(r'\S+@\S+', ' ', txt)
    
    # Remove mentions
    txt = re.sub(r'@\w+', ' ', txt)
    
    # Remove hashtags (keep the word)
    txt = re.sub(r'#(\w+)', r'\1', txt)
    
    # Remove diacritics
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670]')
    txt = arabic_diacritics.sub('', txt)
    
    # Remove tatweel
    txt = txt.replace('\u0640', '')
    
    # Normalize Arabic letters
    txt = re.sub('[إأآٱ]', 'ا', txt)
    txt = txt.replace('ى', 'ي')
    txt = re.sub(r'ة(\s|$)', r'ه\1', txt)
    
    # Remove English and numbers
    txt = re.sub(r'[a-zA-Z0-9]+', ' ', txt)
    
    # Remove punctuation
    txt = re.sub(r'[^\u0600-\u06FF\s]', ' ', txt)
    
    # Collapse repeated characters
    txt = re.sub(r'(.)\1{2,}', r'\1\1', txt)
    
    # Normalize whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    
    return txt

# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

class ToxicTweetClassifier:
    def __init__(self, model_path="arabert_toxic_classifier"):
        """Load the trained model and tokenizer"""
        print("Loading model and tokenizer...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded successfully on {self.device}")
    
    def predict(self, text, show_preprocessing=False):
        """
        Predict if a tweet is toxic or non-toxic
        
        Args:
            text: Raw Arabic text
            show_preprocessing: Show the preprocessed text
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        # Preprocess
        preprocessed = preprocess_text(text)
        
        if show_preprocessing:
            print(f"\nOriginal:     {text}")
            print(f"Preprocessed: {preprocessed}")
        
        # Handle empty text
        if len(preprocessed) == 0:
            return {
                'prediction': 'Non-Toxic',
                'confidence': 0.5,
                'probabilities': {'Non-Toxic': 0.5, 'Toxic': 0.5},
                'note': 'Empty text after preprocessing'
            }
        
        # Tokenize
        inputs = self.tokenizer(
            preprocessed,
            return_tensors="pt",
            max_length=64,
            padding='max_length',
            truncation=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
        
        # Format results
        label = "Toxic" if prediction == 1 else "Non-Toxic"
        confidence = probabilities[prediction].item()
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                'Non-Toxic': probabilities[0].item(),
                'Toxic': probabilities[1].item()
            }
        }
    
    def predict_batch(self, texts):
        """Predict multiple tweets at once"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def interactive_mode():
    """Interactive mode for testing individual tweets"""
    print("\n" + "="*80)
    print("ARABIC TOXIC TWEET CLASSIFIER - INTERACTIVE MODE")
    print("="*80)
    
    classifier = ToxicTweetClassifier()
    
    print("\nInstructions:")
    print("- Enter an Arabic tweet to classify")
    print("- Type 'quit' or 'exit' to stop")
    print("- Type 'examples' to see test examples")
    print("="*80 + "\n")
    
    while True:
        text = input("Enter Arabic tweet: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if text.lower() == 'examples':
            show_examples(classifier)
            continue
        
        if not text:
            print("⚠️  Please enter some text!\n")
            continue
        
        # Predict
        result = classifier.predict(text, show_preprocessing=True)
        
        # Display result
        print("\n" + "-"*80)
        print(f"🔍 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        print(f"📈 Probabilities:")
        print(f"   Non-Toxic: {result['probabilities']['Non-Toxic']*100:.2f}%")
        print(f"   Toxic:     {result['probabilities']['Toxic']*100:.2f}%")
        print("-"*80 + "\n")

def show_examples(classifier):
    """Show predictions on example tweets"""
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    examples = [
        # Non-toxic examples
        ("يوم التأسيس عز وفخر للمملكة السعودية", "Non-Toxic"),
        ("الشعب السعودي ولاؤه لله والحكام والوطن", "Non-Toxic"),
        ("اللهم بارك في يومنا واحفظ بلادنا", "Non-Toxic"),
        
        # Toxic examples  
        ("انت غبي وما تفهم شي", "Toxic"),
        ("كس ام اللي قال كذا", "Toxic"),
        ("يا حمار روح العب بعيد", "Toxic"),
    ]
    
    for i, (text, expected) in enumerate(examples, 1):
        result = classifier.predict(text)
        
        # Emoji for correct/wrong prediction
        emoji = "✅" if result['prediction'] == expected else "❌"
        
        print(f"\n{emoji} Example {i}:")
        print(f"Text:       {text[:80]}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {result['prediction']} ({result['confidence']*100:.1f}%)")
        print(f"Toxic prob: {result['probabilities']['Toxic']*100:.1f}%")
    
    print("\n" + "="*80 + "\n")

def batch_test():
    """Test on a batch of tweets from a file"""
    print("\n" + "="*80)
    print("BATCH TESTING MODE")
    print("="*80)
    
    classifier = ToxicTweetClassifier()
    
    # Example: Test on CSV file
    import pandas as pd
    
    try:
        # Try to load a test file
        filename = input("\nEnter CSV filename (or press Enter for manual input): ").strip()
        
        if not filename:
            # Manual input mode
            print("\nEnter tweets (one per line). Press Enter twice when done:")
            tweets = []
            while True:
                line = input().strip()
                if not line:
                    break
                tweets.append(line)
        else:
            # Load from file
            df = pd.read_csv(filename)
            text_col = [c for c in df.columns if 'text' in c.lower()][0]
            tweets = df[text_col].tolist()[:10]  # First 10 for demo
        
        if not tweets:
            print("No tweets to test!")
            return
        
        print(f"\n📊 Testing {len(tweets)} tweets...\n")
        
        # Predict
        results = classifier.predict_batch(tweets)
        
        # Display results
        toxic_count = sum(1 for r in results if r['prediction'] == 'Toxic')
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY:")
        print(f"Total: {len(tweets)} tweets")
        print(f"Toxic: {toxic_count} ({toxic_count/len(tweets)*100:.1f}%)")
        print(f"Non-Toxic: {len(tweets)-toxic_count} ({(len(tweets)-toxic_count)/len(tweets)*100:.1f}%)")
        print(f"{'='*80}\n")
        
        # Show details
        for i, (tweet, result) in enumerate(zip(tweets, results), 1):
            emoji = "🔴" if result['prediction'] == "Toxic" else "🟢"
            print(f"{emoji} [{i}] {result['prediction']} ({result['confidence']*100:.0f}%)")
            print(f"    {tweet[:60]}...")
            print()
    
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found!")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ARABIC TOXIC TWEET CLASSIFIER")
    print("Model Performance: 99.48% Accuracy | 99.48% F1-Score")
    print("="*80)
    
    print("\nSelect mode:")
    print("1. Interactive mode (test individual tweets)")
    print("2. Batch mode (test multiple tweets)")
    print("3. Show examples")
    print("4. Quick test")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        interactive_mode()
    elif choice == '2':
        batch_test()
    elif choice == '3':
        classifier = ToxicTweetClassifier()
        show_examples(classifier)
    elif choice == '4':
        # Quick test with hardcoded examples
        classifier = ToxicTweetClassifier()
        
        print("\n🚀 Quick Test:\n")
        
        test_cases = [
            "الله يحفظ بلادنا ويحمي شعبنا",
            "انت غبي وما تفهم شي يا حمار",
        ]
        
        for text in test_cases:
            result = classifier.predict(text)
            emoji = "🟢" if result['prediction'] == "Non-Toxic" else "🔴"
            print(f"{emoji} {result['prediction']} ({result['confidence']*100:.0f}%): {text}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()