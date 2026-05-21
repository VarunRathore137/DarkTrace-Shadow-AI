#!/usr/bin/env python3
"""
Create a simple model without custom classes for the streamlit app
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

def create_simple_feature_extractor():
    """Create a simple feature extractor without custom classes"""
    
    # TF-IDF for text features
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100,
        stop_words='english'
    )
    
    # One-hot encoding for categorical features
    onehot = OneHotEncoder(handle_unknown='ignore')
    
    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', tfidf, 'message_text'),
            ('onehot', onehot, ['platform', 'message_type'])
        ],
        remainder='drop'
    )
    
    return preprocessor

def create_simple_model():
    """Create and train a simple model"""
    
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
    print("🔧 Preparing features...")
    
    # Create basic features
    def extract_basic_features(text):
        """Extract basic features from text"""
        text_lower = text.lower()
        
        # Emoji count
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', text))
        
        # Slang count
        slang_words = set(w.lower() for w in slang_dict['slang_term'])
        text_words = set(text_lower.split())
        slang_count = len(text_words.intersection(slang_words))
        
        # Length features
        char_count = len(text)
        word_count = len(text.split())
        
        # Pattern features
        has_price = bool(re.search(r'\$\d+|\d+\s*dollars', text_lower))
        has_quantity = bool(re.search(r'\d+\s*(g|gram|oz|ounce|lb|pound|k|kilo)', text_lower))
        has_location = bool(re.search(r'spot|location|place|meet|behind|alley', text_lower))
        has_urgency = bool(re.search(r'asap|quick|fast|rush|hurry|now|urgent', text_lower))
        
        return [emoji_count, slang_count, char_count, word_count, has_price, has_quantity, has_location, has_urgency]
    
    # Apply feature extraction
    basic_features = conv['message_text'].apply(extract_basic_features)
    basic_features_df = pd.DataFrame(basic_features.tolist(), columns=[
        'emoji_count', 'slang_count', 'char_count', 'word_count', 
        'has_price', 'has_quantity', 'has_location', 'has_urgency'
    ])
    
    # Combine with original data
    X = pd.concat([
        conv[['platform', 'message_type', 'message_text']],
        basic_features_df
    ], axis=1)
    
    # Create labels
    y = (conv['risk_level'].isin(['high', 'medium'])).astype(int)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipeline
    print("🏗️ Creating pipeline...")
    
    # Create feature extractor for basic features
    basic_feature_names = ['emoji_count', 'slang_count', 'char_count', 'word_count', 
                          'has_price', 'has_quantity', 'has_location', 'has_urgency']
    
    def extract_basic_features_transform(X):
        """Transform function for basic features"""
        if isinstance(X, pd.DataFrame):
            return X[basic_feature_names].values
        return X
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'), 'message_text'),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['platform', 'message_type']),
            ('basic', 'passthrough', basic_feature_names)
        ], remainder='drop')),
        ('scaler', StandardScaler(with_mean=False)),  # Fixed for sparse matrices
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import classification_report, roc_auc_score
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    
    print("\n📈 Model Performance:")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")
    
    # Save the model
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    model_path = artifacts_dir / 'simple_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"💾 Model saved to {model_path}")
    
    return pipeline

if __name__ == "__main__":
    print("🔧 Creating simple model for streamlit app...")
    model = create_simple_model()
    
    if model is not None:
        print("✅ Simple model created successfully!")
        print("🎯 You can now use this model in your streamlit app.")
    else:
        print("❌ Failed to create simple model.")
