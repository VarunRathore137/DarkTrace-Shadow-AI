import time
import joblib
import pandas as pd
import numpy as np
import statistics
from pathlib import Path
import pytest
import sys

# Add parent directory to path to import create_final_model
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from create_final_model import create_final_features

def test_inference_sub_millisecond():
    artifacts_dir = ROOT_DIR / "artifacts"
    model_path = artifacts_dir / "xgboost_pipeline.joblib"
    
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
        
    xgb_pipeline = joblib.load(model_path)
    emoji_dict = pd.read_csv(ROOT_DIR / "drug_emoji_dictionary.csv")
    slang_dict = pd.read_csv(ROOT_DIR / "drug_slang_dictionary.csv")
    
    test_msg = "got that fire plug dm me asap, cash only 🔥"
    features = create_final_features(test_msg, emoji_dict, slang_dict)
    
    features_df = pd.DataFrame([features])
    df = pd.DataFrame({
        'platform': ['Telegram'],
        'message_type': ['general'],
        'message_text': [test_msg]
    })
    
    X = pd.concat([df, features_df], axis=1)
    
    # Extract classifier and preprocess data to isolate XGBoost inference time
    preprocessor = xgb_pipeline.named_steps['prep']
    classifier = xgb_pipeline.named_steps['classifier']
    X_transformed = preprocessor.transform(X)
    
    # Warm-up run
    classifier.predict_proba(X_transformed)
    
    runs = 100
    durations = []
    
    for _ in range(runs):
        start_time = time.perf_counter()
        classifier.predict_proba(X_transformed)
        end_time = time.perf_counter()
        # Convert to milliseconds
        durations.append((end_time - start_time) * 1000)
    
    min_ms = min(durations)
    max_ms = max(durations)
    mean_ms = statistics.mean(durations)
    p95_ms = float(np.percentile(durations, 95))
    
    print("\n" + "="*40)
    print(" XGBOOST INFERENCE LATENCY TEST")
    print("="*40)
    print(f"Runs: {runs}")
    print(f"Min:  {min_ms:.4f} ms")
    print(f"Max:  {max_ms:.4f} ms")
    print(f"Mean: {mean_ms:.4f} ms")
    print(f"P95:  {p95_ms:.4f} ms")
    print("="*40)
    
    assert mean_ms < 1.0, f"Mean inference must be sub-millisecond, got {mean_ms:.4f} ms"

