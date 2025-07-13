# backend/ai_symptom_checker.py
import joblib
import numpy as np
from openvino.runtime import Core
import os

# ------------------ MODEL FILES ------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.pkl")
XML_PATH = os.path.join(MODEL_DIR, "symptom_model.xml")
BIN_PATH = os.path.join(MODEL_DIR, "symptom_model.bin")

# ------------------ LOAD COMPONENTS ------------------
vectorizer = joblib.load(VECTORIZER_PATH)
classes = joblib.load(CLASSES_PATH)

ie = Core()
model_ir = ie.read_model(model=XML_PATH, weights=BIN_PATH)
compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# ------------------ INFERENCE FUNCTION ------------------
def predict_condition(symptom_text: str) -> str:
    try:
        X = vectorizer.transform([symptom_text])
        X = X.toarray().astype(np.float32)
        result = compiled_model([X])[output_layer]
        prediction_index = int(np.argmax(result))
        return classes[prediction_index]
    except Exception as e:
        print(f"[SymptomChecker Error] {e}")
        return "Unable to predict condition"
    