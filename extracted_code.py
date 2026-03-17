import pandas as pd
import numpy as np

# ---

df=pd.read_csv('/content/Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv')

# ---

df.head()

# ---

df.shape

# ---

print("Columns:\n", df.columns, "\n")

# ---

print("Missing Values:\n", df.isnull().sum(), "\n")

# ---

print("Emotional State Distribution:\n", df['emotional_state'].value_counts(), "\n")

# ---

# 4. Intensity distribution
print("Intensity Distribution:\n", df['intensity'].value_counts(), "\n")

# ---

# 5. Text length analysis
df['text_length'] = df['journal_text'].astype(str).apply(len)

# ---

print("Text Length Stats:\n", df['text_length'].describe(), "\n")

# ---

print("Short Text Examples:\n", df[df['text_length'] < 20]['journal_text'].head())

# ---

# Fill missing values
df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
df['previous_day_mood'] = df['previous_day_mood'].fillna("unknown")
df['face_emotion_hint'] = df['face_emotion_hint'].fillna("unknown")

# Clean text
df['journal_text'] = df['journal_text'].str.lower().str.strip()

# Add text length feature
df['text_length'] = df['journal_text'].apply(len)

# Encode categorical features
df = pd.get_dummies(df, columns=[
    'ambience_type',
    'time_of_day',
    'previous_day_mood',
    'face_emotion_hint',
    'reflection_quality'
], drop_first=True)

# ---

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)

X_text = tfidf.fit_transform(df['journal_text'])

# ---

df.head()

# ---

df_copy=df.copy()

# ---



# ---

X_meta = df.drop(columns=['id', 'journal_text', 'emotional_state', 'intensity'])
X_meta = X_meta.astype(float)

# ---

from scipy.sparse import hstack
X = hstack([X_text, X_meta])

# ---

y_state = df['emotional_state']
y_intensity = df['intensity']

# ---

from sklearn.linear_model import LogisticRegression

state_model = LogisticRegression(max_iter=1000)
intensity_model = LogisticRegression(max_iter=1000)

# ---

state_model.fit(X, y_state)
intensity_model.fit(X, y_intensity)

# ---

df_test=pd.read_csv('/content/arvyax_test_inputs_120.xlsx - Sheet1.csv')

# ---

df_test.head()

# ---

# Reload original test data (IMPORTANT)


# Fill missing
df_test['sleep_hours'] = df_test['sleep_hours'].fillna(df['sleep_hours'].median())
df_test['previous_day_mood'] = df_test['previous_day_mood'].fillna("unknown")
df_test['face_emotion_hint'] = df_test['face_emotion_hint'].fillna("unknown")

# Clean text
df_test['journal_text'] = df_test['journal_text'].str.lower().str.strip()

# Save text safely
test_text = df_test['journal_text']

# Add text length
df_test['text_length'] = df_test['journal_text'].apply(len)

# One-hot encoding
df_test = pd.get_dummies(df_test, columns=[
    'ambience_type',
    'time_of_day',
    'previous_day_mood',
    'face_emotion_hint',
    'reflection_quality'
], drop_first=True)

# Drop text + id AFTER saving
df_test = df_test.drop(columns=['journal_text', 'id'])

# Align with train
df_test = df_test.reindex(columns=X_meta.columns, fill_value=0)

# ---

df_test.columns

# ---

X_test_text = tfidf.transform(test_text)

from scipy.sparse import hstack
X_test_final = hstack([X_test_text, df_test.astype(float)])

# ---

state_probs = state_model.predict_proba(X_test_final)

# ---

pred_state = state_model.predict(X_test_final)
pred_intensity = intensity_model.predict(X_test_final)

# ---

confidence = state_probs.max(axis=1)
uncertain_flag = (confidence < 0.5).astype(int)

# ---

def decision_engine(row, state, intensity):

    stress = row['stress_level']
    energy = row['energy_level']
    duration = row['duration_min']

    # High stress → immediate calming
    if stress >= 4:
        return "box_breathing", "now"

    # Low energy → rest
    if energy <= 2:
        return "rest", "now"

    # High intensity emotion
    if intensity >= 4:
        return "grounding", "within_15_min"

    # Good energy → productivity
    if energy >= 4 and stress <= 2:
        return "deep_work", "now"

    # Medium case
    if duration < 10:
        return "light_planning", "later_today"

    return "journaling", "tonight"

# ---

actions = []
timings = []

for i, row in df_test.iterrows():
    action, timing = decision_engine(row, pred_state[i], pred_intensity[i])
    actions.append(action)
    timings.append(timing)

# ---

len(actions)

# ---

predictions = pd.DataFrame({
    'id': df_test.index,
    'predicted_state': pred_state,
    'predicted_intensity': pred_intensity,
    'confidence': confidence,
    'uncertain_flag': uncertain_flag,
    'what_to_do': actions,
    'when_to_do': timings
})

predictions.to_csv("predictions.csv", index=False)

# ---

pred_train = state_model.predict(X)

wrong_idx = (pred_train != y_state)
errors = df[wrong_idx]

# ---

errors_sample = errors[['journal_text', 'emotional_state']].head(10)
errors_sample

# ---

import numpy as np

feature_names = tfidf.get_feature_names_out().tolist() + list(X_meta.columns)

importance = np.abs(state_model.coef_).mean(axis=0)

top_idx = np.argsort(importance)[-10:]

for i in top_idx:
    print(feature_names[i])

# ---

import pickle

# Save models
pickle.dump(state_model, open("state_model.pkl", "wb"))
pickle.dump(intensity_model, open("intensity_model.pkl", "wb"))

# Save TF-IDF
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

# Save metadata columns (VERY IMPORTANT)
pickle.dump(X_meta.columns, open("meta_columns.pkl", "wb"))

# ---

state_model = pickle.load(open("state_model.pkl", "rb"))
intensity_model = pickle.load(open("intensity_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
meta_columns = pickle.load(open("meta_columns.pkl", "rb"))

# ---

