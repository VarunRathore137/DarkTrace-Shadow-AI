from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import shap
import sys
import numpy as np
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import create_final_model
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from create_final_model import create_final_features

app = FastAPI(title="Dark Trace AI Backend", version="1.0.0")

# 1. Load ML Artifacts
artifacts_dir = ROOT_DIR / "artifacts"
try:
    xgb_pipeline = joblib.load(artifacts_dir / "xgboost_pipeline.joblib")
    preprocessor = joblib.load(artifacts_dir / "xgboost_preprocessor.joblib")
    classifier = xgb_pipeline.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    emoji_dict = pd.read_csv(ROOT_DIR / "drug_emoji_dictionary.csv")
    slang_dict = pd.read_csv(ROOT_DIR / "drug_slang_dictionary.csv")
    print("✅ ML Models and dictionaries loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ML models or dictionaries: {e}")
    xgb_pipeline = None

# 2. Load RAG Dependencies
try:
    print("🚀 Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path=str(ROOT_DIR / "chroma_data"))
    try:
        chroma_collection = chroma_client.get_collection(name="drug_conversations")
        print("✅ ChromaDB and SentenceTransformer loaded successfully.")
    except Exception:
        print("⚠️ Chroma collection 'drug_conversations' not found.")
        chroma_collection = None
except Exception as e:
    print(f"❌ Error loading RAG components: {e}")
    embedding_model = None
    chroma_client = None
    chroma_collection = None

class MessageInput(BaseModel):
    message: str

class BatchMessageInput(BaseModel):
    messages: List[str]

class SemanticSearchInput(BaseModel):
    query: str
    limit: Optional[int] = 5
    platform: Optional[str] = None
    risk_level: Optional[str] = None

@app.get("/health")
def health():
    status = "ok" if xgb_pipeline is not None else "degraded"
    return {"status": status}

def process_messages(messages: List[str]):
    feature_list = []
    for msg in messages:
        features = create_final_features(msg, emoji_dict, slang_dict)
        feature_list.append(features)
        
    features_df = pd.DataFrame(feature_list)
    
    df = pd.DataFrame({
        'platform': ['Telegram'] * len(messages),
        'message_type': ['general'] * len(messages),
        'message_text': messages
    })
    
    X = pd.concat([df, features_df], axis=1)
    return X, features_df

@app.post("/predict")
def predict(input_data: MessageInput, explain: bool = Query(False, description="Whether to include SHAP explanations")):
    if xgb_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly.")
        
    X, features_df = process_messages([input_data.message])
    
    proba = float(xgb_pipeline.predict_proba(X)[0, 1])
    risk_level = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    
    response = {
        "prediction": proba,
        "risk_level": risk_level,
        "features": features_df.iloc[0].to_dict()
    }
    
    if explain:
        try:
            X_transformed = preprocessor.transform(X)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
                
            shap_values = explainer.shap_values(X_transformed)
            
            if isinstance(shap_values, list):
                shap_array = shap_values[1][0].tolist() 
            else:
                shap_array = shap_values[0].tolist()
                
            response["shap_values"] = shap_array
            
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)
                
            response["shap_base_value"] = base_value
        except Exception as e:
            response["shap_error"] = str(e)
    
    return response

@app.post("/batch_predict")
def batch_predict(input_data: BatchMessageInput):
    if xgb_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly.")
        
    X, _ = process_messages(input_data.messages)
    
    probas = xgb_pipeline.predict_proba(X)[:, 1]
    
    results = []
    for proba in probas:
        proba = float(proba)
        risk_level = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
        results.append({
            "prediction": proba,
            "risk_level": risk_level
        })
        
    return {"results": results}

@app.post("/semantic_search")
def semantic_search(input_data: SemanticSearchInput):
    if chroma_collection is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="ChromaDB or embedding model not loaded.")

    try:
        query_embedding = embedding_model.encode([input_data.query]).tolist()
        
        # Build where filter
        where_filter = {}
        if input_data.platform and input_data.risk_level:
            where_filter = {
                "$and": [
                    {"platform": input_data.platform},
                    {"risk_level": input_data.risk_level}
                ]
            }
        elif input_data.platform:
            where_filter = {"platform": input_data.platform}
        elif input_data.risk_level:
            where_filter = {"risk_level": input_data.risk_level}

        query_params = {
            "query_embeddings": query_embedding,
            "n_results": input_data.limit,
            "include": ["metadatas", "documents", "distances"]
        }
        
        if where_filter:
            query_params["where"] = where_filter

        results = chroma_collection.query(**query_params)
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i]
                
                # Convert L2 distance to a 0-1 similarity score roughly
                similarity_score = 1.0 / (1.0 + float(dist))
                
                formatted_results.append({
                    "similarity_score": similarity_score,
                    "message_text": doc,
                    "platform": meta.get("platform"),
                    "risk_level": meta.get("risk_level"),
                    "timestamp": meta.get("timestamp"),
                    "message_id": meta.get("message_id")
                })

        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
