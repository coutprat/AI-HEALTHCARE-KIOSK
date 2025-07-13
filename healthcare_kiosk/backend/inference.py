from fastapi import FastAPI
from pydantic import BaseModel
from openvino.runtime import Core
import numpy as np
import joblib
import os
from symptom_extractor import extract_symptom_keywords

app = FastAPI(title="AI Symptom Checker - Free Text Inference")

# === Define paths ===
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
IR_MODEL_PATH = os.path.join(MODEL_DIR, "symptom_model.xml")
ENCODER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "classes.pkl")

# === Load OpenVINO model ===
core = Core()
model = core.read_model(IR_MODEL_PATH)
compiled_model = core.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()

# === Load vectorizer and label encoder ===
vectorizer = joblib.load(ENCODER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# === Input schema ===
class SymptomText(BaseModel):
    user_input: str

from typing import Dict, Any, Union

class PredictionResponse(BaseModel):
    prediction: str
    extracted_symptoms: list[str]
    user_input: str

class ErrorResponse(BaseModel):
    error: str

@app.post("/predict", response_model=Union[PredictionResponse, ErrorResponse])
def predict_from_text(symptom_text: SymptomText) -> Union[PredictionResponse, ErrorResponse]:
    text = symptom_text.user_input.strip()
    
    # Extract symptoms using NLP
    extracted_symptoms = extract_symptom_keywords(text)
    
    if not extracted_symptoms:
        return ErrorResponse(error="No valid symptoms found in input.")

    # Convert extracted symptoms to input vector
    joined_text = " ".join(extracted_symptoms)
    input_vector = vectorizer.transform([joined_text]).toarray().astype(np.float32)

    # Perform inference
    result = infer_request.infer({0: input_vector})
    output = next(iter(result.values()))
    pred_idx = int(np.argmax(output))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return PredictionResponse(
        prediction=pred_label,
        extracted_symptoms=extracted_symptoms,
        user_input=text
    )
