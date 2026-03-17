# 🧠 ArvyaX Machine Learning Internship Assignment  
### Theme: From Understanding Humans → To Guiding Them

---

## 🚀 Project Overview

This project builds an AI system that goes beyond prediction to:

- Understand human emotional state  
- Reason under noisy and imperfect inputs  
- Decide meaningful next actions  
- Guide users toward better mental states  

The system takes user reflections and contextual signals as input and outputs:
- Emotional state  
- Intensity  
- Recommended action (what to do)  
- Suggested timing (when to do it)  
- Confidence + uncertainty awareness  

---

## ⚙️ Setup Instructions

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn scipy
```

### 2. Run the notebook / script
- Open the notebook in Jupyter or Google Colab  
- Run all cells step-by-step  

---

## 🧠 Approach

### 1. Data Preprocessing
- Filled missing values:
  - `sleep_hours` → median  
  - categorical → "unknown"  
- Cleaned text (lowercase, strip)
- One-hot encoded categorical features  
- Added `text_length` feature  

---

### 2. Feature Engineering
- **Text Features:** TF-IDF vectorization  
- **Metadata Features:** sleep, stress, energy, time, etc.  
- Combined using sparse matrix  

---

### 3. Model Selection

- **Emotional State:** Classification (Logistic Regression)  
- **Intensity:** Classification (1–5 scale)  

Reason:
- Lightweight  
- Fast  
- Suitable for edge deployment  

---

### 4. Decision Engine (Core Logic)

Rule-based system using:
- predicted state  
- intensity  
- stress level  
- energy level  
- time of day  

Outputs:
- `what_to_do` (e.g., breathing, rest, deep work)  
- `when_to_do` (now, later, tonight, etc.)  

---

### 5. Uncertainty Modeling

- Confidence derived from prediction probabilities  
- `uncertain_flag = 1` when:
  - low confidence  
  - short text  
  - conflicting signals  

---

## 📊 Feature Importance

- Text features (keywords like *distracted, calm, drained*) are strong indicators  
- Metadata (stress, energy, sleep) improves context understanding  
- Best performance achieved by combining both  

---

## ⚖️ Ablation Study

| Model | Performance |
|------|------------|
| Text Only | Good |
| Text + Metadata | Better |

### Insight:
- Text captures emotion  
- Metadata resolves ambiguity  

---

## ❗ Error Analysis

Key failure cases:
- Short text ("ok", "felt heavy")  
- Conflicting emotions  
- Noisy labels  
- Incomplete/truncated inputs  

### Improvements:
- Use uncertainty flags  
- Incorporate better embeddings  
- Handle label noise  

---

## 📱 Edge Deployment Plan

- Model: TF-IDF + Logistic Regression  
- Lightweight and fast  
- Runs fully offline  
- Low memory usage  

### Trade-offs:
- Faster but less expressive than large models  

---

## 📂 Project Structure

```
project/
│
├── notebook.ipynb
├── state_model.pkl
├── intensity_model.pkl
├── tfidf.pkl
├── meta_columns.pkl
├── predictions.csv
├── README.md
├── ERROR_ANALYSIS.md
├── EDGE_PLAN.md
```

---

## ▶️ How to Run

1. Load models using pickle  
2. Preprocess input (same as training)  
3. Convert to features (TF-IDF + metadata)  
4. Predict:
   - emotional state  
   - intensity  
5. Apply decision engine  
6. Generate final output  

---

## 🎯 Key Highlights

- Handles noisy and imperfect real-world data  
- Combines ML + rule-based reasoning  
- Includes uncertainty awareness  
- Designed for edge deployment  
- Focuses on meaningful user guidance  

---

## 🧠 Conclusion

This system demonstrates a practical approach to building intelligent systems that not only predict but also guide users. It balances machine learning with reasoning and is designed to operate in real-world, imperfect conditions.

---