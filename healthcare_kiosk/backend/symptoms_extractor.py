import os
import pandas as pd
from rapidfuzz import process, fuzz
import re

# Load symptom list from CSV
BASE_DIR = os.path.dirname(__file__)
SYMPTOM_CSV_PATH = os.path.join(BASE_DIR, "healthcare_kiosk", "symptoms_disease.csv")

# Load all symptoms into memory
ALL_SYMPTOMS = []
try:
    if os.path.exists(SYMPTOM_CSV_PATH):
        df = pd.read_csv(SYMPTOM_CSV_PATH)
        symptom_set = set()
        for symptoms in df['symptoms']:
            if pd.notna(symptoms):  # Check for NaN values
                for sym in str(symptoms).split('|'):
                    clean_sym = sym.strip().lower().replace('_', ' ')
                    if clean_sym:
                        symptom_set.add(clean_sym)
        ALL_SYMPTOMS = list(symptom_set)
        print(f"[INFO] Loaded {len(ALL_SYMPTOMS)} symptoms from CSV")
    else:
        print(f"[WARNING] {SYMPTOM_CSV_PATH} not found")
except Exception as e:
    print(f"[ERROR] Failed to load symptoms from CSV: {e}")

# Expanded symptom list as fallback
KNOWN_SYMPTOMS = [
    "fever", "cough", "headache", "sore throat", "shortness of breath", "chest pain",
    "fatigue", "vomiting", "diarrhea", "rash", "body pain", "joint pain", "dizziness",
    "nausea", "runny nose", "loss of taste", "loss of smell", "blurred vision", "weakness",
    "cold", "anxiety", "depression", "palpitations", "itching", "burning sensation",
    "stomach pain", "back pain", "muscle pain", "swelling", "constipation", "insomnia",
    "sweating", "chills", "difficulty swallowing", "hoarseness", "ear pain", "sneezing",
    "dry cough", "breathlessness", "arm pain", "nasal congestion", "watery eyes",
    "itchy eyes", "sensitivity to light", "aura", "wheezing", "chest tightness",
    "pale skin", "restlessness", "panic", "phlegm", "chest discomfort", "weight gain",
    "cold intolerance", "dry skin", "heat intolerance", "tremors", "weight loss",
    "postnasal drip", "facial pain", "pressure", "burning urination", "cloudy urine",
    "frequent urination", "neck stiffness", "confusion", "high fever", "thirst"
]

# Combine all symptoms
ALL_SYMPTOMS_COMBINED = list(set(ALL_SYMPTOMS + KNOWN_SYMPTOMS))

def extract_symptom_keywords(text: str):
    """Extract symptoms from user input text"""
    if not text:
        return []
    
    text = text.lower().strip()
    extracted = set()
    
    # Clean text - remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Direct keyword matching
    for symptom in ALL_SYMPTOMS_COMBINED:
        if symptom in text:
            extracted.add(symptom)
    
    # Phrase matching for multi-word symptoms
    words = text.split()
    for i in range(len(words)):
        for j in range(i+1, min(i+4, len(words)+1)):  # Check up to 3-word phrases
            phrase = ' '.join(words[i:j])
            for symptom in ALL_SYMPTOMS_COMBINED:
                if phrase == symptom or (len(phrase) > 3 and phrase in symptom):
                    extracted.add(symptom)
    
    # Fuzzy matching for individual words
    for word in words:
        if len(word) > 3:  # Only match words longer than 3 characters
            try:
                match, score, _ = process.extractOne(word, ALL_SYMPTOMS_COMBINED, scorer=fuzz.ratio)
                if score >= 85:  # 85% similarity threshold
                    extracted.add(match)
            except Exception as e:
                print(f"[WARNING] Fuzzy matching error for '{word}': {e}")
    
    result = list(extracted)
    print(f"[INFO] Extracted symptoms: {result}")
    return result

def extract_symptoms(user_input: str, threshold: int = 85):
    """Alternative function for backward compatibility"""
    return extract_symptom_keywords(user_input)
