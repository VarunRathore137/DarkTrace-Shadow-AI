#!/usr/bin/env python3
"""
Create the final XGBoost model for the streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from create_final_model import create_final_features

def create_xgboost_model():
    """Create and train the XGBoost model"""
    
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
    
    feature_list = []
    for text in conv['message_text']:
        features = create_final_features(text, emoji_dict, slang_dict)
        feature_list.append(features)
        
    features_df = pd.DataFrame(feature_list)
    
    X = pd.concat([
        conv[['platform', 'message_type', 'message_text']],
        features_df
    ], axis=1)
    
    y = (conv['risk_level'].isin(['high', 'medium'])).astype(int)
    
    print(f"📈 Dataset shape: {X.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🏗️ Creating final, optimized XGBoost pipeline...")
    
    feature_columns = [col for col in features_df.columns if col not in ['message_text', 'platform', 'message_type', 'risk_category']]
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2)), 'message_text'),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['platform', 'message_type']),
            ('features', 'passthrough', feature_columns)
        ], remainder='drop')),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    print("🚀 Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    print("\n📈 Model Performance:")
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")
    
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save entire pipeline
    model_path = artifacts_dir / 'xgboost_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Final model saved to {model_path}")
    
    # Also save the native XGBoost model
    xgb_model = pipeline.named_steps['classifier']
    native_model_path = artifacts_dir / 'xgboost_model.json'
    xgb_model.save_model(native_model_path)
    print(f"💾 Native XGBoost model saved to {native_model_path}")

    # Save the preprocessor
    preprocessor = Pipeline(pipeline.steps[:-1])
    preprocessor_path = artifacts_dir / 'xgboost_preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"💾 Preprocessor saved to {preprocessor_path}")

    # Save feature info
    accuracy = np.mean(y_pred == y_val)
    feature_info = {
        'feature_columns': feature_columns,
        'model_type': 'XGBoost',
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_accuracy': float(accuracy)
    }
    
    feature_path = artifacts_dir / 'xgboost_feature_info.joblib'
    joblib.dump(feature_info, feature_path)
    
    return pipeline

if __name__ == "__main__":
    create_xgboost_model()
