# backend/vitals.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# Mock DB
vitals_data = []

router = APIRouter(prefix="/vitals", tags=["Vitals"])

# -----------------------------
# Pydantic Schema
# -----------------------------
class VitalInput(BaseModel):
    user_id: int
    temperature: float = Field(..., gt=90, lt=110)
    heart_rate: int = Field(..., gt=30, lt=200)
    systolic_bp: int = Field(..., gt=80, lt=200)
    diastolic_bp: int = Field(..., gt=40, lt=120)
    oxygen_saturation: float = Field(..., ge=70.0, le=100.0)

class VitalRecord(VitalInput):
    timestamp: datetime

def get_vitals():
    # Dummy vitals data for testing
    return {
        "heart_rate": "72 bpm",
        "blood_pressure": "120/80 mmHg",
        "temperature": "98.6 °F",
        "oxygen_saturation": "98%"
    }

# -----------------------------
# Routes
# -----------------------------
@router.post("/submit", response_model=VitalRecord)
def submit_vitals(vitals: VitalInput):
    record = VitalRecord(**vitals.dict(), timestamp=datetime.utcnow())
    vitals_data.append(record)
    return record


@router.get("/history/{user_id}", response_model=List[VitalRecord])
def get_vital_history(user_id: int):
    history = [v for v in vitals_data if v.user_id == user_id]
    if not history:
        raise HTTPException(status_code=404, detail="No vitals found for this user")
    return history
