# backend/ehr.py

from sqlalchemy.orm import Session
from backend.database import models, crud, database

def get_patient_ehr(patient_name: str):
    db: Session = database.SessionLocal()
    try:
        # Fetch patient record
        patient = crud.get_patient_by_name(db, name=patient_name)
        if not patient:
            return {"error": "Patient not found."}

        # Fetch vitals and diagnosis
        vitals = crud.get_latest_vitals(db, patient_id=patient.id)
        diagnosis = crud.get_latest_diagnosis(db, patient_id=patient.id)

        # Compose EHR summary
        ehr_summary = {
            "Patient Name": patient.name,
            "Age": patient.age,
            "Gender": patient.gender,
            "Vitals": {
                "Temperature": vitals.temperature if vitals else "N/A",
                "Heart Rate": vitals.heart_rate if vitals else "N/A",
                "Blood Pressure": vitals.blood_pressure if vitals else "N/A",
            },
            "Diagnosis": diagnosis.condition if diagnosis else "N/A",
            "Notes": diagnosis.notes if diagnosis else "N/A",
            "Date": diagnosis.timestamp.isoformat() if diagnosis else "N/A"
        }

        return ehr_summary

    finally:
        db.close()
