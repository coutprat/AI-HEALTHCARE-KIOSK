from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from . import models
from .models import User
from datetime import datetime
from typing import Optional, Dict, Any

# -------------------- Patient --------------------

def create_patient(db: Session, aadhaar: str, name: str, age: int, gender: str):
    try:
        # Check if patient already exists
        existing = db.query(models.Patient).filter_by(aadhaar=aadhaar).first()
        if existing:
            print(f"[ERROR] Patient with Aadhaar {aadhaar} already exists")
            return None
        
        # Create new patient
        patient = models.Patient(aadhaar=aadhaar, name=name, age=age, gender=gender)
        db.add(patient)
        db.commit()
        db.refresh(patient)
        print(f"[SUCCESS] Created patient: {patient.name} (ID: {patient.id})")
        return patient
    
    except IntegrityError as e:
        db.rollback()
        print(f"[ERROR] Database integrity error: {e}")
        return None
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to create patient: {e}")
        return None

def get_patient_by_aadhaar(db: Session, aadhaar: str):
    try:
        patient = db.query(models.Patient).filter_by(aadhaar=aadhaar).first()
        if patient:
            print(f"[SUCCESS] Found patient: {patient.name}")
        else:
            print(f"[INFO] No patient found with Aadhaar: {aadhaar}")
        return patient
    except Exception as e:
        print(f"[ERROR] Failed to retrieve patient: {e}")
        return None

def get_patient_by_name(db: Session, name: str):
    try:
        return db.query(models.Patient).filter(models.Patient.name == name).first()
    except Exception as e:
        print(f"[ERROR] Failed to retrieve patient by name: {e}")
        return None

# -------------------- Vitals --------------------

def add_vitals(db: Session, patient_id: int, height: float, weight: float, bp: str, pulse: int):
    try:
        # Verify patient exists
        patient = db.query(models.Patient).filter_by(id=patient_id).first()
        if not patient:
            print(f"[ERROR] Patient with ID {patient_id} not found")
            return None
        
        # Calculate BMI
        bmi = weight / (height/100)**2
        
        vitals = models.Vitals(
            patient_id=patient_id,
            height_cm=height,
            weight_kg=weight,
            blood_pressure=bp,
            pulse=pulse,
            bmi=round(bmi, 2)
        )
        
        db.add(vitals)
        db.commit()
        db.refresh(vitals)
        print(f"[SUCCESS] Added vitals for patient ID {patient_id}")
        return vitals
    
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to add vitals: {e}")
        return None

# -------------------- Diagnosis --------------------

def add_diagnosis(db: Session, patient_id: int, symptoms: str, result: str):
    try:
        # Verify patient exists
        patient = db.query(models.Patient).filter_by(id=patient_id).first()
        if not patient:
            print(f"[ERROR] Patient with ID {patient_id} not found")
            return None
        
        diagnosis = models.Diagnosis(
            patient_id=patient_id,
            symptoms=symptoms,
            result=result
        )
        
        db.add(diagnosis)
        db.commit()
        db.refresh(diagnosis)
        print(f"[SUCCESS] Added diagnosis for patient ID {patient_id}")
        return diagnosis
    
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to add diagnosis: {e}")
        return None

# -------------------- Full EHR --------------------

def get_patient_ehr(db: Session, patient_id: int) -> Optional[Dict[str, Any]]:
    try:
        patient = db.query(models.Patient).filter_by(id=patient_id).first()
        if not patient:
            print(f"[ERROR] Patient with ID {patient_id} not found")
            return None
        
        ehr_data = {
            "patient": {
                "id": patient.id,
                "aadhaar": patient.aadhaar,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender
            },
            "vitals": [
                {
                    "id": v.id,
                    "height": v.height_cm,
                    "weight": v.weight_kg,
                    "bp": v.blood_pressure,
                    "pulse": v.pulse,
                    "bmi": v.bmi,
                    "timestamp": v.timestamp.isoformat() if v.timestamp else None
                } for v in patient.vitals
            ],
            "diagnoses": [
                {
                    "id": d.id,
                    "symptoms": d.symptoms,
                    "result": d.result,
                    "timestamp": d.timestamp.isoformat() if d.timestamp else None
                } for d in patient.diagnoses
            ]
        }
        
        print(f"[SUCCESS] Retrieved EHR for patient: {patient.name}")
        return ehr_data
    
    except Exception as e:
        print(f"[ERROR] Failed to retrieve EHR: {e}")
        return None

# -------------------- User --------------------

def get_user_by_username(db: Session, username: str):
    try:
        return db.query(User).filter(User.username == username).first()
    except Exception as e:
        print(f"[ERROR] Failed to retrieve user: {e}")
        return None

def create_user(db: Session, username: str, hashed_password: str, email: str, full_name: str):
    try:
        new_user = User(
            username=username,
            hashed_password=hashed_password,
            email=email,
            full_name=full_name,
            created_at=datetime.utcnow()
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"[SUCCESS] Created user: {username}")
        return new_user
    
    except IntegrityError as e:
        db.rollback()
        print(f"[ERROR] User creation failed - integrity error: {e}")
        return None
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to create user: {e}")
        return None
