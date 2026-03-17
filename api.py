import pickle
import pandas as pd
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="ArvyaX Emotional Support API")

# 1. Load the trained Sklearn models statically (loads once on startup)
state_model = pickle.load(open("state_model.pkl", "rb"))
intensity_model = pickle.load(open("intensity_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
meta_columns = pickle.load(open("meta_columns.pkl", "rb"))

# 2. Load Lightweight Conversational Model (SLM)
# Using google/flan-t5-small because it's only ~300MB, runs great on CPU, and is good for generating short empathic text.
slm = pipeline("text2text-generation", model="google/flan-t5-small")

class ReflectionInput(BaseModel):
    reflection: str
    duration_min: int = 15
    sleep_hours: float = 7.0
    energy_level: int = 3
    stress_level: int = 3
    ambience_type: str = "none"
    time_of_day: str = "none"
    previous_day_mood: str = "none"
    face_emotion_hint: str = "none"
    reflection_quality: str = "none"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "ArvyaX API running and ready."}

@app.post("/analyze")
def analyze_reflection(req: ReflectionInput):
    # Prepare input numerical variables
    input_data = {
        'duration_min': req.duration_min,
        'sleep_hours': req.sleep_hours,
        'energy_level': req.energy_level,
        'stress_level': req.stress_level,
        'text_length': len(req.reflection.strip().lower()),
    }
    
    # Prepare the one-hot encoded features
    cat_features = {
        f"ambience_type_{req.ambience_type}": 1.0,
        f"time_of_day_{req.time_of_day}": 1.0,
        f"previous_day_mood_{req.previous_day_mood}": 1.0,
        f"face_emotion_hint_{req.face_emotion_hint}": 1.0,
        f"reflection_quality_{req.reflection_quality}": 1.0
    }
    
    # Reconstruct the feature matrix to exactly match what it learned on during training
    meta_df = pd.DataFrame(columns=meta_columns)
    meta_df.loc[0] = 0.0 # Setup a row of zeros
    
    for k, v in input_data.items():
        if k in meta_df.columns:
            meta_df.at[0, k] = float(v)
            
    for k, v in cat_features.items():
        if k in meta_df.columns:
            meta_df.at[0, k] = 1.0
            
    # Process text
    x_text = tfidf.transform([req.reflection.lower().strip()])
    x_final = hstack([x_text, meta_df.astype(float)])
    
    # Predict emotional class and intensity
    pred_state = state_model.predict(x_final)[0]
    pred_intensity = intensity_model.predict(x_final)[0]
    
    # Find confidence
    state_probs = state_model.predict_proba(x_final)[0]
    confidence = float(state_probs.max())
    
    # 3. Generate supportive message using SLM
    prompt = f"Offer a short, deeply empathic, and comforting response to someone feeling '{pred_state}' who says: '{req.reflection}'"
    res = slm(prompt, max_length=50, num_return_sequences=1)
    supportive_msg = res[0]['generated_text']
    
    return {
        "pred_state": pred_state,
        "pred_intensity": int(pred_intensity),
        "confidence": round(confidence * 100, 2),
        "supportive_message": supportive_msg
    }
