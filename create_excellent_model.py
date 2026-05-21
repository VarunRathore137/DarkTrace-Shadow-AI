#!/usr/bin/env python3
"""
Create an excellent, highly reliable model for the streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def create_excellent_features(text, emoji_dict, slang_dict):
    """Create highly discriminative features"""
    text_lower = text.lower()
    words = text_lower.split()
    
    features = {}
    
    # 1. Enhanced Emoji Analysis
    emojis = re.findall(r'[\U0001F300-\U0001F9FF]', text)
    features['emoji_count'] = len(emojis)
    
    # Drug-related emojis (highly suspicious)
    drug_emojis = {'🍁', '💊', '💉', '🔥', '💨', '🌿', '🌱', '🪴', '🍄', '🔌', '💊', '💉', '💊', '💊', '💊', '💊', '💊', '💊', '💊', '💊'}
    drug_emoji_count = sum(1 for e in emojis if e in drug_emojis)
    features['drug_emoji_count'] = drug_emoji_count
    features['has_drug_emoji'] = drug_emoji_count > 0
    features['drug_emoji_ratio'] = drug_emoji_count / max(len(emojis), 1)
    
    # 2. Enhanced Slang Analysis
    slang_words = set(w.lower() for w in slang_dict['slang_term'])
    text_slang = [w for w in words if w in slang_words]
    features['slang_count'] = len(text_slang)
    features['has_slang'] = len(text_slang) > 0
    features['slang_ratio'] = len(text_slang) / max(len(words), 1)
    
    # 3. Text Statistics
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    # 4. High-Risk Pattern Detection
    # Price patterns (very suspicious)
    price_patterns = [
        r'\$\d+',  # $50
        r'\d+\s*dollars?',  # 50 dollars
        r'\d+\s*bucks?',  # 50 bucks
        r'price.*\d+',  # price 50
        r'cost.*\d+',  # cost 50
        r'\d+\s*per\s*(?:g|gram|oz|ounce|lb|pound)',  # 50 per gram
        r'\d+\s*for\s*(?:g|gram|oz|ounce|lb|pound)'  # 50 for gram
    ]
    features['has_price'] = any(re.search(pattern, text_lower) for pattern in price_patterns)
    
    # Quantity patterns (very suspicious)
    quantity_patterns = [
        r'\d+\s*g(?:ram)?',  # 5g, 5 gram
        r'\d+\s*oz(?:ounce)?',  # 5oz, 5 ounce
        r'\d+\s*lb(?:s)?',  # 5lb, 5 pounds
        r'\d+\s*k(?:ilo)?',  # 5k, 5 kilo
        r'\d+\s*piece',  # 5 piece
        r'\d+\s*pack',  # 5 pack
        r'\d+\s*bag',  # 5 bag
        r'\d+\s*sheet'  # 5 sheet
    ]
    features['has_quantity'] = any(re.search(pattern, text_lower) for pattern in quantity_patterns)
    
    # Location patterns (suspicious)
    location_patterns = [
        r'spot', r'location', r'place', r'meet', r'behind', r'alley', 
        r'corner', r'parking', r'back', r'private', r'discrete', r'usual',
        r'secure', r'safe', r'quiet', r'low\s*key'
    ]
    features['has_location'] = any(re.search(pattern, text_lower) for pattern in location_patterns)
    
    # Urgency patterns (suspicious)
    urgency_patterns = [
        r'asap', r'quick', r'fast', r'rush', r'hurry', r'now', 
        r'urgent', r'immediate', r'right now', r'tonight', r'limited time',
        r'going fast', r'won\'t last', r'last chance'
    ]
    features['has_urgency'] = any(re.search(pattern, text_lower) for pattern in urgency_patterns)
    
    # 5. Behavioral Patterns
    # Discretion patterns (highly suspicious)
    discretion_patterns = [
        r'quiet', r'private', r'secret', r'discrete', r'careful', 
        r'safe', r'clean', r'low key', r'under radar', r'no questions',
        r'cash only', r'no trace'
    ]
    features['has_discretion'] = any(re.search(pattern, text_lower) for pattern in discretion_patterns)
    
    # Quality patterns (suspicious)
    quality_patterns = [
        r'good', r'pure', r'clean', r'best', r'quality', r'fire', 
        r'premium', r'top', r'grade a', r'pure', r'lab tested',
        r'authentic', r'genuine', r'real deal'
    ]
    features['has_quality'] = any(re.search(pattern, text_lower) for pattern in quality_patterns)
    
    # Transaction patterns (highly suspicious)
    transaction_patterns = [
        r'have', r'got', r'available', r'supply', r'plug', r'connect', 
        r'hook', r'deal', r'offer', r'stock', r'in stock', r'fresh batch',
        r'just landed', r'new arrival', r'limited supply'
    ]
    features['has_transaction'] = any(re.search(pattern, text_lower) for pattern in transaction_patterns)
    
    # 6. Advanced Suspicious Patterns
    # Crypto/payment patterns
    payment_patterns = [
        r'crypto', r'btc', r'bitcoin', r'cash', r'venmo', r'paypal', 
        r'zelle', r'cash app', r'venmo', r'no card', r'cash only'
    ]
    features['has_payment'] = any(re.search(pattern, text_lower) for pattern in payment_patterns)
    
    # Time patterns
    time_patterns = [
        r'tonight', r'tomorrow', r'weekend', r'friday', r'saturday', 
        r'after hours', r'late night', r'midnight', r'now', r'asap'
    ]
    features['has_time'] = any(re.search(pattern, text_lower) for pattern in time_patterns)
    
    # 7. Highly Suspicious Combinations (weighted heavily)
    features['suspicious_combo_1'] = features['has_drug_emoji'] and features['has_price']
    features['suspicious_combo_2'] = features['has_drug_emoji'] and features['has_quantity']
    features['suspicious_combo_3'] = features['has_slang'] and features['has_location']
    features['suspicious_combo_4'] = features['has_price'] and features['has_quantity']
    features['suspicious_combo_5'] = features['has_urgency'] and features['has_discretion']
    
    # 8. Risk Score (weighted heuristic)
    high_risk_factors = [
        features['has_drug_emoji'] * 3,  # Weight heavily
        features['has_slang'] * 2,
        features['has_price'] * 2,
        features['has_quantity'] * 2,
        features['has_location'] * 1,
        features['has_urgency'] * 1,
        features['has_discretion'] * 2,
        features['has_quality'] * 1,
        features['has_transaction'] * 2,
        features['has_payment'] * 1,
        features['suspicious_combo_1'] * 4,  # Very heavily weighted
        features['suspicious_combo_2'] * 4,
        features['suspicious_combo_3'] * 3,
        features['suspicious_combo_4'] * 3,
        features['suspicious_combo_5'] * 3
    ]
    features['weighted_risk_score'] = sum(high_risk_factors)
    
    # 9. Text complexity (normal messages tend to be more complex)
    features['text_complexity'] = len(set(words)) / max(len(words), 1)  # Type-token ratio
    features['has_punctuation'] = bool(re.search(r'[.!?]', text))
    features['has_capitalization'] = bool(re.search(r'[A-Z]', text))
    
    return features

def create_excellent_model():
    """Create and train an excellent model"""
    
    # Load data
    conv_path = Path('synthetic_drug_conversations.csv')
    emoji_dict_path = Path('drug_emoji_dictionary.csv')
    slang_dict_path = Path('drug_slang_dictionary.csv')
    
    if not conv_path.exists():
        print("❌ synthetic_drug_conversations.csv not found!")
        return None
    
    print("📊 Loading data...")
    conv = pd.read_csv(conv_path)
    emoji_dict = pd.read_csv(emoji_dict_path)
    slang_dict = pd.read_csv(slang_dict_path)
    
    # Prepare features
    print("🔧 Creating excellent features...")
    
    # Create excellent features
    feature_list = []
    for text in conv['message_text']:
        features = create_excellent_features(text, emoji_dict, slang_dict)
        feature_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    
    # Combine with original data
    X = pd.concat([
        conv[['platform', 'message_type', 'message_text']],
        features_df
    ], axis=1)
    
    # Create labels
    y = (conv['risk_level'].isin(['high', 'medium'])).astype(int)
    
    print(f"📈 Dataset shape: {X.shape}")
    print(f"🎯 Label distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📊 Training data: {y_train.value_counts().to_dict()}")
    print(f"📊 Validation data: {y_val.value_counts().to_dict()}")
    
    # Create pipeline
    print("🏗️ Creating excellent pipeline...")
    
    # Get feature names for the model
    feature_columns = [col for col in features_df.columns if col not in ['message_text', 'platform', 'message_type']]
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 3)), 'message_text'),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['platform', 'message_type']),
            ('features', 'passthrough', feature_columns)
        ], remainder='drop')),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    
    # Train the model
    print("🚀 Training excellent model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Model Performance:")
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")
    
    # Test on examples with expected results
    print("\n🧪 Testing on examples:")
    test_messages = [
        ("Hello, how are you today?", "Normal"),
        ("got that good work 💊, $30 per half, dm", "Suspicious"),
        ("Let's meet for coffee tomorrow", "Normal"),
        ("fresh batch just landed, limited time only 🔥", "Suspicious"),
        ("What's the weather like?", "Normal"),
        ("need that 🔌 asap, got cash", "Suspicious"),
        ("How was your weekend?", "Normal"),
        ("quality stuff available, dm for menu", "Suspicious"),
        ("Can you help me with homework?", "Normal"),
        ("🔥 pure grade, $100 per g, meet at usual spot", "Suspicious")
    ]
    
    for msg, expected in test_messages:
        test_df = pd.DataFrame({
            'message_text': [msg],
            'platform': ['telegram'],
            'message_type': ['general']
        })
        
        # Add features
        features = create_excellent_features(msg, emoji_dict, slang_dict)
        for key, value in features.items():
            test_df[key] = [value]
        
        try:
            proba = pipeline.predict_proba(test_df)[0, 1]
            risk_level = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW"
            status = "✅" if (proba >= 0.4 and expected == "Suspicious") or (proba < 0.4 and expected == "Normal") else "❌"
            print(f"{status} '{msg[:50]}...' -> {proba:.1%} ({risk_level}) - Expected: {expected}")
        except Exception as e:
            print(f"❌ Error with message: {e}")
    
    # Save the model
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    model_path = artifacts_dir / 'excellent_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Excellent model saved to {model_path}")
    
    # Save feature names for reference
    feature_info = {
        'feature_columns': feature_columns,
        'total_features': len(feature_columns) + 300 + 20,  # TF-IDF + One-hot + custom features
        'model_type': 'GradientBoosting',
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'smote_applied': False # SMOTE is removed
    }
    
    feature_path = artifacts_dir / 'excellent_feature_info.joblib'
    joblib.dump(feature_info, feature_path)
    print(f"💾 Feature info saved to {feature_path}")
    
    return pipeline

if __name__ == "__main__":
    print("🔧 Creating excellent, highly reliable model for streamlit app...")
    model = create_excellent_model()
    
    if model is not None:
        print("✅ Excellent model created successfully!")
        print("🎯 This model should provide highly accurate predictions.")
        print("🚀 You can now use this model in your streamlit app.")
    else:
        print("❌ Failed to create excellent model.")
