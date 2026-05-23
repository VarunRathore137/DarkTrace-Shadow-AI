import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import re
import uuid
import datetime
from tqdm import tqdm
import json
import warnings

# Suppress some noisy warnings from transformers
warnings.filterwarnings('ignore')

BATCH_SIZE = 500
CHROMA_PATH = "./chroma_data"
CSV_PATH = "synthetic_drug_conversations.csv"
CHECKPOINT_FILE = "ingest_checkpoint.json"

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Remove unicode replacement character (often caused by broken encoding)
    text = text.replace('\ufffd', '')
    # Remove weird spaces / extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f).get("last_index", 0)
        except json.JSONDecodeError:
            pass
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_index": index}, f)

def main():
    print("🚀 Initializing ChromaDB and SentenceTransformer...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="drug_conversations")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"📊 Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    start_idx = load_checkpoint()
    total_rows = len(df)
    
    if start_idx >= total_rows:
        print("✅ Ingestion already complete according to checkpoint.")
        return

    print(f"🔄 Starting ingestion from index {start_idx} / {total_rows}")
    
    # Process in batches
    for i in tqdm(range(start_idx, total_rows, BATCH_SIZE)):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        
        ids = []
        documents = []
        metadatas = []
        
        for _, row in batch_df.iterrows():
            text = normalize_text(row.get('message_text', ''))
            if not text:
                continue
                
            doc_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            
            ids.append(doc_id)
            documents.append(text)
            
            # Metadata filtering setup
            metadatas.append({
                "platform": str(row.get('platform', 'unknown')),
                "risk_level": str(row.get('risk_level', 'unknown')),
                "message_id": doc_id,
                "timestamp": timestamp
            })
            
        if documents:
            embeddings = model.encode(documents).tolist()
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        save_checkpoint(i + len(batch_df))
        
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()
