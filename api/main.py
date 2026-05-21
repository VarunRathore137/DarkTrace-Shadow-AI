from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Dark Trace AI Backend", version="1.0.0")

class MessageInput(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: MessageInput):
    message_lower = input_data.message.lower()
    
    # Simple heuristics to make prediction interactive
    score = 0.05
    if any(k in message_lower for k in ["weed", "pill", "buy", "sell", "cash", "price", "meet", "plug"]):
        score += 0.4
    if any(e in input_data.message for e in ["🍁", "💊", "💉", "🔥", "💨", "🌿", "🔌"]):
        score += 0.5
    
    score = min(score, 1.0)
    risk_level = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
    
    # Constructing mock feature details
    # Streamlit uses:
    # emoji_count, drug_emoji_count, slang_count, has_price, has_quantity, has_location, has_urgency, has_punctuation, has_capitalization
    emojis = [c for c in input_data.message if ord(c) > 127] # very simple emoji approximation
    drug_emojis = [e for e in emojis if e in ["🍁", "💊", "💉", "🔥", "💨", "🌿", "🔌"]]
    
    features = {
        "emoji_count": len(emojis),
        "drug_emoji_count": len(drug_emojis),
        "has_drug_emoji": len(drug_emojis) > 0,
        "slang_count": 1 if "weed" in message_lower or "plug" in message_lower else 0,
        "has_slang": "weed" in message_lower or "plug" in message_lower,
        "has_price": "$" in message_lower or "price" in message_lower,
        "has_quantity": "g" in message_lower or "gram" in message_lower,
        "has_location": "meet" in message_lower or "spot" in message_lower,
        "has_urgency": "asap" in message_lower or "now" in message_lower,
        "char_count": len(input_data.message),
        "word_count": len(input_data.message.split()),
        "has_punctuation": any(c in input_data.message for c in ".!?"),
        "has_capitalization": any(c.isupper() for c in input_data.message)
    }
    
    return {
        "prediction": score,
        "risk_level": risk_level,
        "features": features
    }
