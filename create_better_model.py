#!/usr/bin/env python3
"""
Create a better, more reliable model for the streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_better_features(text, emoji_dict, slang_dict):
    """Create comprehensive features for better detection"""
    text_lower = text.lower()
    words = text_lower.split()
    
    features = {}
    
    # 1. Emoji Analysis
    emojis = re.findall(r'[\U0001F300-\U0001F9FF]', text)
    features['emoji_count'] = len(emojis)
    features['emoji_density'] = len(emojis) / max(len(words), 1)
    
    # Drug-related emojis
    drug_emojis = {'🍁', '💊', '💉', '🔥', '💨', '🌿', '🌱', '🪴', '🍄', '🔌', '💊', '💉', '💊', '💊'}
    drug_emoji_count = sum(1 for e in emojis if e in drug_emojis)
    features['drug_emoji_count'] = drug_emoji_count
    features['has_drug_emoji'] = drug_emoji_count > 0
    
    # 2. Slang Analysis
    slang_words = set(w.lower() for w in slang_dict['slang_term'])
    text_slang = [w for w in words if w in slang_words]
    features['slang_count'] = len(text_slang)
    features['slang_density'] = len(text_slang) / max(len(words), 1)
    features['has_slang'] = len(text_slang) > 0
    
    # 3. Text Statistics
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['char_word_ratio'] = len(text) / max(len(words), 1)
    
    # 4. Suspicious Patterns
    # Price patterns
    price_patterns = [
        r'\$\d+',  # $50
        r'\d+\s*dollars?',  # 50 dollars
        r'\d+\s*bucks?',  # 50 bucks
        r'price.*\d+',  # price 50
        r'cost.*\d+'  # cost 50
    ]
    features['has_price'] = any(re.search(pattern, text_lower) for pattern in price_patterns)
    
    # Quantity patterns
    quantity_patterns = [
        r'\d+\s*g(?:ram)?',  # 5g, 5 gram
        r'\d+\s*oz(?:ounce)?',  # 5oz, 5 ounce
        r'\d+\s*lb(?:s)?',  # 5lb, 5 pounds
        r'\d+\s*k(?:ilo)?',  # 5k, 5 kilo
        r'\d+\s*piece',  # 5 piece
        r'\d+\s*pack'  # 5 pack
    ]
    features['has_quantity'] = any(re.search(pattern, text_lower) for pattern in quantity_patterns)
    
    # Location patterns
    location_patterns = [
        r'spot', r'location', r'place', r'meet', r'behind', r'alley', 
        r'corner', r'parking', r'back', r'private', r'discrete'
    ]
    features['has_location'] = any(re.search(pattern, text_lower) for pattern in location_patterns)
    
    # Urgency patterns
    urgency_patterns = [
        r'asap', r'quick', r'fast', r'rush', r'hurry', r'now', 
        r'urgent', r'immediate', r'right now', r'tonight'
    ]
    features['has_urgency'] = any(re.search(pattern, text_lower) for pattern in urgency_patterns)
    
    # 5. Behavioral Patterns
    # Discretion patterns
    discretion_patterns = [
        r'quiet', r'private', r'secret', r'discrete', r'careful', 
        r'safe', 'clean', r'low key', r'under radar'
    ]
    features['has_discretion'] = any(re.search(pattern, text_lower) for pattern in discretion_patterns)
    
    # Quality patterns
    quality_patterns = [
        r'good', r'pure', r'clean', r'best', r'quality', r'fire', 
        r'premium', r'top', r'grade a', r'pure'
    ]
    features['has_quality'] = any(re.search(pattern, text_lower) for pattern in quality_patterns)
    
    # Transaction patterns
    transaction_patterns = [
        r'have', r'got', r'available', r'supply', r'plug', r'connect', 
        r'hook', r'deal', r'offer', r'stock'
    ]
    features['has_transaction'] = any(re.search(pattern, text_lower) for pattern in transaction_patterns)
    
    # 6. Advanced Patterns
    # Crypto/payment patterns
    payment_patterns = [
        r'crypto', r'btc', r'bitcoin', r'cash', r'venmo', r'paypal', 
        r'zelle', r'cash app', r'venmo', r'zeecash'
    ]
    features['has_payment'] = any(re.search(pattern, text_lower) for pattern in payment_patterns)
    
    # Time patterns
    time_patterns = [
        r'tonight', r'tomorrow', r'weekend', r'friday', r'saturday', 
        r'after hours', r'late night', r'midnight', r'sun down'
    ]
    features['has_time'] = any(re.search(pattern, text_lower) for pattern in time_patterns)
    
    # 7. Suspicious Combinations
    features['suspicious_combo'] = (
        (features['has_price'] and features['has_quantity']) or
        (features['has_drug_emoji'] and features['has_price']) or
        (features['has_slang'] and features['has_location']) or
        (features['has_urgency'] and features['has_discretion'])
    )
    
    # 8. Risk Score (simple heuristic)
    risk_factors = [
        features['has_drug_emoji'],
        features['has_slang'],
        features['has_price'],
        features['has_quantity'],
        features['has_location'],
        features['has_urgency'],
        features['has_discretion'],
        features['has_quality'],
        features['has_transaction'],
        features['has_payment'],
        features['suspicious_combo']
    ]
    features['risk_factors'] = sum(risk_factors)
    
    return features

def create_better_model():
    """Create and train a better model"""
    
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
    print("🔧 Creating comprehensive features...")
    
    # Create better features
    feature_list = []
    for text in conv['message_text']:
        features = create_better_features(text, emoji_dict, slang_dict)
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
    
    # Create pipeline
    print("🏗️ Creating advanced pipeline...")
    
    # Get feature names for the model
    feature_columns = [col for col in features_df.columns if col not in ['message_text', 'platform', 'message_type']]
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2)), 'message_text'),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['platform', 'message_type']),
            ('features', 'passthrough', feature_columns)
        ], remainder='drop')),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Train the model
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Model Performance:")
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")
    
    # Test on some examples
    print("\n🧪 Testing on examples:")
    test_messages = [
        "Hello, how are you today?",  # Normal
        "got that good work 💊, $30 per half, dm",  # Suspicious
        "Let's meet for coffee tomorrow",  # Normal
        "fresh batch just landed, limited time only 🔥",  # Suspicious
        "What's the weather like?",  # Normal
    ]
    
    for msg in test_messages:
        test_df = pd.DataFrame({
            'message_text': [msg],
            'platform': ['telegram'],
            'message_type': ['general']
        })
        
        # Add features
        features = create_better_features(msg, emoji_dict, slang_dict)
        for key, value in features.items():
            test_df[key] = [value]
        
        try:
            proba = pipeline.predict_proba(test_df)[0, 1]
            risk_level = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW"
            print(f"'{msg[:50]}...' -> {proba:.1%} ({risk_level})")
        except Exception as e:
            print(f"Error with message: {e}")
    
    # Save the model
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    model_path = artifacts_dir / 'better_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Better model saved to {model_path}")
    
    # Save feature names for reference
    feature_info = {
        'feature_columns': feature_columns,
        'total_features': len(feature_columns) + 200 + 20,  # TF-IDF + One-hot + custom features
        'model_type': 'RandomForest',
        'training_samples': len(X_train),
        'validation_samples': len(X_val)
    }
    
    feature_path = artifacts_dir / 'feature_info.joblib'
    joblib.dump(feature_info, feature_path)
    print(f"💾 Feature info saved to {feature_path}")
    
    return pipeline

if __name__ == "__main__":
    print("🔧 Creating better, more reliable model for streamlit app...")
    model = create_better_model()
    
    if model is not None:
        print("✅ Better model created successfully!")
        print("🎯 This model should provide much more accurate predictions.")
        print("🚀 You can now use this model in your streamlit app.")
    else:
        print("❌ Failed to create better model.")
