#!/usr/bin/env python3
"""
Create the final, highly optimized model for the streamlit app
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

def create_final_features(text, emoji_dict, slang_dict):
    """Create highly discriminative features with better thresholds"""
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
    
    # 4. High-Risk Pattern Detection (with better patterns)
    # Price patterns (very suspicious)
    price_patterns = [
        r'\$\d+',  # $50
        r'\d+\s*dollars?',  # 50 dollars
        r'\d+\s*bucks?',  # 50 bucks
        r'price.*\d+',  # price 50
        r'cost.*\d+',  # cost 50
        r'\d+\s*per\s*(?:g|gram|oz|ounce|lb|pound)',  # 50 per gram
        r'\d+\s*for\s*(?:g|gram|oz|ounce|lb|pound)',  # 50 for gram
        r'\d+\s*each',  # 50 each
        r'\d+\s*apiece'  # 50 apiece
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
        r'\d+\s*sheet',  # 5 sheet
        r'\d+\s*tab',  # 5 tab
        r'\d+\s*pill'  # 5 pill
    ]
    features['has_quantity'] = any(re.search(pattern, text_lower) for pattern in quantity_patterns)
    
    # Location patterns (suspicious)
    location_patterns = [
        r'spot', r'location', r'place', r'meet', r'behind', r'alley', 
        r'corner', r'parking', r'back', r'private', r'discrete', r'usual',
        r'secure', r'safe', r'quiet', r'low\s*key', r'discreet'
    ]
    features['has_location'] = any(re.search(pattern, text_lower) for pattern in location_patterns)
    
    # Urgency patterns (suspicious)
    urgency_patterns = [
        r'asap', r'quick', r'fast', r'rush', r'hurry', r'now', 
        r'urgent', r'immediate', r'right now', r'tonight', r'limited time',
        r'going fast', r'won\'t last', r'last chance', r'hit me up',
        r'hmu', r'dm me', r'message me'
    ]
    features['has_urgency'] = any(re.search(pattern, text_lower) for pattern in urgency_patterns)
    
    # 5. Behavioral Patterns
    # Discretion patterns (highly suspicious)
    discretion_patterns = [
        r'quiet', r'private', r'secret', r'discrete', r'careful', 
        r'safe', r'clean', r'low key', r'under radar', r'no questions',
        r'cash only', r'no trace', r'discreet', r'confidential'
    ]
    features['has_discretion'] = any(re.search(pattern, text_lower) for pattern in discretion_patterns)
    
    # Quality patterns (suspicious)
    quality_patterns = [
        r'good', r'pure', r'clean', r'best', r'quality', r'fire', 
        r'premium', r'top', r'grade a', r'pure', r'lab tested',
        r'authentic', r'genuine', r'real deal', r'legit', r'real'
    ]
    features['has_quality'] = any(re.search(pattern, text_lower) for pattern in quality_patterns)
    
    # Transaction patterns (highly suspicious)
    transaction_patterns = [
        r'have', r'got', r'available', r'supply', r'plug', r'connect', 
        r'hook', r'deal', r'offer', r'stock', r'in stock', r'fresh batch',
        r'just landed', r'new arrival', r'limited supply', r'available now',
        r'ready to go', r'good to go'
    ]
    features['has_transaction'] = any(re.search(pattern, text_lower) for pattern in transaction_patterns)
    
    # 6. Advanced Suspicious Patterns
    # Crypto/payment patterns
    payment_patterns = [
        r'crypto', r'btc', r'bitcoin', r'cash', r'venmo', r'paypal', 
        r'zelle', r'cash app', r'venmo', r'no card', r'cash only',
        r'cash only', r'no trace'
    ]
    features['has_payment'] = any(re.search(pattern, text_lower) for pattern in payment_patterns)
    
    # Time patterns
    time_patterns = [
        r'tonight', r'tomorrow', r'weekend', r'friday', r'saturday', 
        r'after hours', r'late night', r'midnight', r'now', r'asap',
        r'right now', r'immediate'
    ]
    features['has_time'] = any(re.search(pattern, text_lower) for pattern in time_patterns)
    
    # 7. Highly Suspicious Combinations (weighted heavily)
    features['suspicious_combo_1'] = features['has_drug_emoji'] and features['has_price']
    features['suspicious_combo_2'] = features['has_drug_emoji'] and features['has_quantity']
    features['suspicious_combo_3'] = features['has_slang'] and features['has_location']
    features['suspicious_combo_4'] = features['has_price'] and features['has_quantity']
    features['suspicious_combo_5'] = features['has_urgency'] and features['has_discretion']
    features['suspicious_combo_6'] = features['has_drug_emoji'] and features['has_slang']
    features['suspicious_combo_7'] = features['has_price'] and features['has_location']
    
    # 8. Risk Score (weighted heuristic)
    high_risk_factors = [
        features['has_drug_emoji'] * 4,  # Weight very heavily
        features['has_slang'] * 3,
        features['has_price'] * 3,
        features['has_quantity'] * 3,
        features['has_location'] * 2,
        features['has_urgency'] * 2,
        features['has_discretion'] * 3,
        features['has_quality'] * 2,
        features['has_transaction'] * 3,
        features['has_payment'] * 2,
        features['suspicious_combo_1'] * 5,  # Very heavily weighted
        features['suspicious_combo_2'] * 5,
        features['suspicious_combo_3'] * 4,
        features['suspicious_combo_4'] * 4,
        features['suspicious_combo_5'] * 4,
        features['suspicious_combo_6'] * 4,
        features['suspicious_combo_7'] * 4
    ]
    features['weighted_risk_score'] = sum(high_risk_factors)
    
    # 9. Text complexity (normal messages tend to be more complex)
    features['text_complexity'] = len(set(words)) / max(len(words), 1)  # Type-token ratio
    features['has_punctuation'] = bool(re.search(r'[.!?]', text))
    features['has_capitalization'] = bool(re.search(r'[A-Z]', text))
    features['has_question'] = bool(re.search(r'\?', text))
    features['has_exclamation'] = bool(re.search(r'!', text))
    
    # 10. Normal message indicators (negative features)
    normal_patterns = [
        r'hello', r'hi', r'hey', r'how are you', r'good morning',
        r'good afternoon', r'good evening', r'thank you', r'thanks',
        r'please', r'excuse me', r'sorry', r'goodbye', r'bye',
        r'see you', r'take care', r'have a good day'
    ]
    features['has_normal_greeting'] = any(re.search(pattern, text_lower) for pattern in normal_patterns)
    
    # 11. Final risk assessment
    if features['weighted_risk_score'] >= 8:
        features['risk_category'] = 'HIGH'
    elif features['weighted_risk_score'] >= 4:
        features['risk_category'] = 'MEDIUM'
    else:
        features['risk_category'] = 'LOW'
    
    return features

def create_final_model():
    """Create and train the final, highly optimized model"""
    
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
    print("🔧 Creating final, optimized features...")
    
    # Create final features
    feature_list = []
    for text in conv['message_text']:
        features = create_final_features(text, emoji_dict, slang_dict)
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
    print("🏗️ Creating final, optimized pipeline...")
    
    # Get feature names for the model
    feature_columns = [col for col in features_df.columns if col not in ['message_text', 'platform', 'message_type', 'risk_category']]
    
    # Create the final pipeline
    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2)), 'message_text'),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['platform', 'message_type']),
            ('features', 'passthrough', feature_columns)
        ], remainder='drop')),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Train the model
    print("🚀 Training final, optimized model...")
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
        ("🔥 pure grade, $100 per g, meet at usual spot", "Suspicious"),
        ("Good morning! How are you?", "Normal"),
        ("$50 per gram, meet at the usual place", "Suspicious"),
        ("Have a great day!", "Normal"),
        ("need 🔌 asap, got the cash ready", "Suspicious")
    ]
    
    correct_predictions = 0
    total_predictions = len(test_messages)
    
    for msg, expected in test_messages:
        test_df = pd.DataFrame({
            'message_text': [msg],
            'platform': ['telegram'],
            'message_type': ['general']
        })
        
        # Add features
        features = create_final_features(msg, emoji_dict, slang_dict)
        for key, value in features.items():
            if key != 'risk_category':  # Don't include the risk category in features
                test_df[key] = [value]
        
        try:
            proba = pipeline.predict_proba(test_df)[0, 1]
            risk_level = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW"
            
            # Check if prediction matches expectation
            is_correct = False
            if expected == "Suspicious" and proba >= 0.4:
                is_correct = True
            elif expected == "Normal" and proba < 0.4:
                is_correct = True
            
            if is_correct:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} '{msg[:50]}...' -> {proba:.1%} ({risk_level}) - Expected: {expected}")
        except Exception as e:
            print(f"❌ Error with message: {e}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\n🎯 Test Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Save the model
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    model_path = artifacts_dir / 'final_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Final model saved to {model_path}")
    
    # Save feature names for reference
    feature_info = {
        'feature_columns': feature_columns,
        'total_features': len(feature_columns) + 200 + 20,  # TF-IDF + One-hot + custom features
        'model_type': 'RandomForest',
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_accuracy': accuracy
    }
    
    feature_path = artifacts_dir / 'final_feature_info.joblib'
    joblib.dump(feature_info, feature_path)
    print(f"💾 Feature info saved to {feature_path}")
    
    return pipeline

if __name__ == "__main__":
    print("🔧 Creating final, highly optimized model for streamlit app...")
    model = create_final_model()
    
    if model is not None:
        print("✅ Final model created successfully!")
        print("🎯 This model should provide highly accurate predictions.")
        print("🚀 You can now use this model in your streamlit app.")
    else:
        print("❌ Failed to create final model.")
