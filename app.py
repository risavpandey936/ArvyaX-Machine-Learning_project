import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack

# Set up page config
st.set_page_config(page_title="ArvyaX Emotional Support AI", page_icon="🧠", layout="wide")

# Custom CSS for a beautiful design
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTextInput>div>div>input {
        color: #c9d1d9;
    }
    .model-header {
        font-family: 'Inter', sans-serif;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(48, 54, 61, 0.4);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .metric-title {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .action-card {
        background: linear-gradient(135deg, #1f6feb 0%, #238636 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    state_model = pickle.load(open("state_model.pkl", "rb"))
    intensity_model = pickle.load(open("intensity_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    meta_columns = pickle.load(open("meta_columns.pkl", "rb"))
    return state_model, intensity_model, tfidf, meta_columns

@st.cache_resource(show_spinner="Loading Conversational AI Model (First boot up takes ~10s)...")
def load_slm():
    from transformers import pipeline
    # Use google/flan-t5-small because it handles conversational text well, runs locally, 
    # and has a tiny (~300MB) footprint suitable for minimal CPU inference.
    slm = pipeline("text2text-generation", model="google/flan-t5-small")
    return slm

def decision_engine(stress, energy, duration, intensity, state):
    # High stress → immediate calming
    if stress >= 4:
        return "Box Breathing", "Now"
    # Low energy → rest
    if energy <= 2:
        return "Rest", "Now"
    # High intensity emotion
    if intensity >= 4:
        return "Grounding Exercises", "Within 15 Min"
    # Good energy → productivity
    if energy >= 4 and stress <= 2:
        return "Deep Work", "Now"
    # Medium case
    if duration < 10:
        return "Light Planning", "Later Today"
    
    return "Journaling", "Tonight"

st.markdown('<h1 class="model-header">🧠 ArvyaX Human Understanding Engine</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>Understanding emotional states and guiding you towards better mental wellbeing.</p>", unsafe_allow_html=True)

try:
    state_model, intensity_model, tfidf, meta_columns = load_models()
except Exception as e:
    st.error("Failed to load models. Make sure pkl files exist in the same directory.")
    st.stop()

with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Your Reflection")
        
        reflection_options = [
            "Write your own reflection...",
            "I've been feeling quite drained and overwhelmed by my tasks lately.",
            "I'm feeling very calm and focused today, ready to tackle my work.",
            "I feel extremely anxious and stressed about an upcoming deadline.",
            "Honestly, I'm just tired. I haven't been sleeping well and lack energy.",
            "I'm feeling pretty neutral today. Not too happy, not too sad."
        ]
        selected_option = st.selectbox("Quick Options:", reflection_options)
        default_text = "" if selected_option == "Write your own reflection..." else selected_option
        
        journal_text = st.text_area("How are you feeling right now? What's on your mind?", value=default_text, height=150, placeholder="E.g., I've been feeling quite drained and overwhelmed by my tasks lately...")
        
        st.subheader("📊 Biomarkers & Context")
        c1, c2 = st.columns(2)
        duration_min = c1.number_input("Session duration (min)", value=15, min_value=1)
        sleep_hours = c2.number_input("Sleep hours", value=7.0, step=0.5)
        
        energy_level = st.slider("Energy Level (1-5)", 1, 5, 3)
        stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)

    with col2:
        st.subheader("🌍 Environmental Variables")
        ambience_type = st.selectbox("Ambience", ["cafe", "forest", "mountain", "ocean", "rain"])
        time_of_day = st.selectbox("Time of Day", ["afternoon", "early_morning", "morning", "evening", "night"])
        previous_day_mood = st.selectbox("Previous Day Mood", ["calm", "focused", "mixed", "neutral", "overwhelmed", "restless", "unknown"])
        face_emotion_hint = st.selectbox("Facial Emotion Hint", ["sad_face", "happy_face", "neutral_face", "none", "tense_face", "tired_face", "unknown"])
        reflection_quality = st.selectbox("Reflection Quality", ["confident", "conflicted", "vague"])

    predict_btn = st.button("Analyze & Guide 🚀", use_container_width=True, type="primary")

if predict_btn and journal_text.strip() != "":
    with st.spinner("Analyzing emotional indicators..."):
        # Process inputs
        input_data = {
            'duration_min': duration_min,
            'sleep_hours': sleep_hours,
            'energy_level': energy_level,
            'stress_level': stress_level,
            'text_length': len(journal_text.strip().lower()),
        }
        
        # Categoricals - we add the prefix to match meta_columns
        cat_features = {
            f"ambience_type_{ambience_type}": 1.0,
            f"time_of_day_{time_of_day}": 1.0,
            f"previous_day_mood_{previous_day_mood}": 1.0,
            f"face_emotion_hint_{face_emotion_hint}": 1.0,
            f"reflection_quality_{reflection_quality}": 1.0
        }
        
        # Create DataFrame aligned with meta_columns
        meta_df = pd.DataFrame(columns=meta_columns)
        meta_df.loc[0] = 0.0 # initialize with 0
        
        # Fill numeric values
        for k, v in input_data.items():
            if k in meta_df.columns:
                meta_df.at[0, k] = float(v)
                
        # Fill hot-encoded values
        for k, v in cat_features.items():
            if k in meta_df.columns:
                meta_df.at[0, k] = 1.0
                
        # Transform text
        x_text = tfidf.transform([journal_text.lower().strip()])
        
        # Combine
        x_final = hstack([x_text, meta_df.astype(float)])
        
        # Predict
        pred_state = state_model.predict(x_final)[0]
        pred_intensity = intensity_model.predict(x_final)[0]
        
        state_probs = state_model.predict_proba(x_final)[0]
        confidence = state_probs.max()
        
        # Decision logic
        action, timing = decision_engine(stress_level, energy_level, duration_min, pred_intensity, pred_state)
        
        st.markdown("---")
        st.markdown("### 🎯 Analysis Results")
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Emotional State</div>
                <div class="metric-value">{pred_state.title()}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with r2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Intensity Level</div>
                <div class="metric-value">{pred_intensity} / 5</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with r3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Confidence</div>
                <div class="metric-value">{confidence*100:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Supportive SLM Chat Generation
        try:
            slm = load_slm()
            prompt = f"Offer a comforting, very short empathic sentence to someone feeling '{pred_state}' who said: '{journal_text}'"
            slm_response = slm(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
            
            st.markdown(f'''
            <div style="background-color: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                <h3 style="color: #58a6ff; margin-top: 0px; font-size: 1.1rem;">💬 Companion AI Notice</h3>
                <p style="margin-bottom: 0px; font-style: italic;">"{slm_response}"</p>
            </div>
            ''', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Failed to load conversational AI for support message: {e}")
        
        st.markdown(f'''
        <div class="action-card">
            <h2 style="color: white; margin-bottom: 5px;">Recommended Action</h2>
            <div style="font-size: 24px; font-weight: bold;">{action}</div>
            <div style="margin-top: 5px; opacity: 0.9;">Best time to do this: <b>{timing}</b></div>
        </div>
        ''', unsafe_allow_html=True)
        
        if confidence < 0.5:
            st.warning("⚠️ The model has low confidence in this prediction due to conflicting signals or short text input. Please use your best judgment.")
            
elif predict_btn:
    st.error("Please enter some text in your reflection first.")
