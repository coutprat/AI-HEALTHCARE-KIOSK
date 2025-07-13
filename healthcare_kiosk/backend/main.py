from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from .database import models, crud, database
from .database.database import SessionLocal, init_db
from . import ai_symptom_checker
from .symptoms_extractor import extract_symptom_keywords
from pydantic import BaseModel, validator
import os
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize DB - this will create tables
print("Initializing database...")
try:
    db_path = init_db()
    if db_path and os.path.exists(db_path):
        print(f"✅ Database file created at: {db_path}")
    else:
        print(f"⚠️ Database initialized but path verification failed")
        # Set a default path for verification
        db_path = "./backend/database/healthcare.db"
        if os.path.exists(db_path):
            print(f"✅ Database file found at: {db_path}")
        else:
            print(f"❌ Database file not found at expected location")
            # Create a fallback path
            db_path = "healthcare.db"
except Exception as e:
    print(f"❌ Database initialization error: {e}")
    db_path = "./backend/database/healthcare.db"

# FastAPI app
app = FastAPI(
    title="Intel Healthcare Kiosk Backend",
    description="Backend for Aadhaar-Enabled Healthcare Services with AI Symptom Checker",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {exc.body}")
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "body": exc.body.decode() if exc.body else None
        }
    )

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for request/response validation
class PatientCreate(BaseModel):
    aadhaar: str
    name: str
    age: int
    gender: str
    
    @validator('aadhaar')
    def validate_aadhaar(cls, v):
        if not v or len(v) != 12:
            raise ValueError('Aadhaar must be exactly 12 digits')
        if not v.isdigit():
            raise ValueError('Aadhaar must contain only digits')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @validator('age')
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError('Age must be between 1 and 120')
        return v
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female', 'Other']:
            raise ValueError('Gender must be Male, Female, or Other')
        return v

class VitalsCreate(BaseModel):
    patient_id: int
    height: float
    weight: float
    bp: str
    pulse: int

class SymptomInput(BaseModel):
    user_input: str
    patient_id: int

class DiagnosisResponse(BaseModel):
    diagnosis: str
    extracted_symptoms: List[str]
    user_input: str
    patient_id: int
    diagnosis_id: int

class LoginResponse(BaseModel):
    patient_id: int
    name: str
    age: int
    gender: str
    message: str

# ------------------ Root Endpoint ------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Intel Healthcare Kiosk API",
        "status": "Running",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /register - Register new patient",
            "login": "GET /login?aadhaar={aadhaar} - Login with Aadhaar",
            "vitals": "POST /vitals - Add patient vitals",
            "diagnose": "POST /diagnose - AI symptom diagnosis",
            "ehr": "GET /ehr?patient_id={id} - Get patient EHR",
            "test-db": "GET /test-db - Test database connectivity",
            "docs": "GET /docs - API documentation"
        }
    }

# ------------------ Patient Registration ------------------
@app.post("/register")
def register_patient(patient_data: PatientCreate, db: Session = Depends(get_db)):
    """Register a new patient with detailed error logging"""
    try:
        logger.info(f"Registration request received: {patient_data}")
        
        # Check if patient already exists
        existing_patient = crud.get_patient_by_aadhaar(db, patient_data.aadhaar)
        if existing_patient:
            logger.error(f"Patient already exists with Aadhaar: {patient_data.aadhaar}")
            raise HTTPException(status_code=400, detail="Patient with this Aadhaar already registered")
        
        # Create new patient
        patient = crud.create_patient(
            db, 
            patient_data.aadhaar, 
            patient_data.name, 
            patient_data.age, 
            patient_data.gender
        )
        
        if not patient:
            logger.error("Failed to create patient in database")
            raise HTTPException(status_code=500, detail="Failed to create patient")
        
        logger.info(f"Patient created successfully: {patient.id}")
        return {
            "message": "Patient registered successfully",
            "patient_id": patient.id,
            "name": patient.name,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ------------------ Patient Login ------------------
@app.get("/login", response_model=LoginResponse)
def login_patient(aadhaar: str, db: Session = Depends(get_db)):
    """Login patient using Aadhaar number"""
    try:
        if len(aadhaar) != 12:
            raise HTTPException(status_code=400, detail="Aadhaar must be exactly 12 digits")
        
        patient = crud.get_patient_by_aadhaar(db, aadhaar)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found. Please register first.")
        
        return LoginResponse(
            patient_id=patient.id,
            name=patient.name,
            age=patient.age,
            gender=patient.gender,
            message=f"Welcome back, {patient.name}!"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# ------------------ Patient Login by Name ------------------
@app.get("/login-by-name", response_model=LoginResponse)
def login_by_name(name: str, db: Session = Depends(get_db)):
    """Login patient using name (for face recognition)"""
    try:
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        
        # Search for patient by name
        patient = crud.get_patient_by_name(db, name.strip())
        if not patient:
            raise HTTPException(status_code=404, detail=f"No patient found with name: {name}")
        
        logger.info(f"Face recognition login successful for: {patient.name}")
        
        return LoginResponse(
            patient_id=patient.id,
            name=patient.name,
            age=patient.age,
            gender=patient.gender,
            message=f"Welcome back, {patient.name}! (Face Recognition Login)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Name-based login failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# ------------------ Vitals Management ------------------
@app.post("/vitals")
def add_patient_vitals(vitals_data: VitalsCreate, db: Session = Depends(get_db)):
    """Add vital signs for a patient"""
    try:
        patient = db.query(models.Patient).filter_by(id=vitals_data.patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        if vitals_data.height <= 0 or vitals_data.weight <= 0:
            raise HTTPException(status_code=400, detail="Height and weight must be positive values")
        
        if vitals_data.pulse <= 0:
            raise HTTPException(status_code=400, detail="Pulse must be a positive value")
        
        vitals = crud.add_vitals(
            db,
            vitals_data.patient_id,
            vitals_data.height,
            vitals_data.weight,
            vitals_data.bp,
            vitals_data.pulse
        )
        
        if not vitals:
            raise HTTPException(status_code=500, detail="Failed to add vitals")
        
        bmi_category = "Normal"
        if vitals.bmi < 18.5:
            bmi_category = "Underweight"
        elif vitals.bmi >= 25 and vitals.bmi < 30:
            bmi_category = "Overweight"
        elif vitals.bmi >= 30:
            bmi_category = "Obese"
        
        return {
            "message": "Vitals recorded successfully",
            "vitals_id": vitals.id,
            "patient_name": patient.name,
            "bmi": vitals.bmi,
            "bmi_category": bmi_category,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adding vitals failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Adding vitals failed: {str(e)}")

# ------------------ AI Symptom Checker ------------------
@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose_symptoms(symptom_input: SymptomInput, db: Session = Depends(get_db)):
    """AI-powered symptom analysis and disease prediction"""
    try:
        patient = db.query(models.Patient).filter_by(id=symptom_input.patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        if not symptom_input.user_input or not symptom_input.user_input.strip():
            raise HTTPException(status_code=400, detail="Symptom description cannot be empty")
        
        logger.info(f"Processing symptoms for patient: {patient.name}")
        logger.info(f"User input: {symptom_input.user_input}")
        
        extracted_symptoms = extract_symptom_keywords(symptom_input.user_input)
        
        if not extracted_symptoms:
            raise HTTPException(
                status_code=400, 
                detail="No valid symptoms found in input. Please describe your symptoms more clearly."
            )
        
        logger.info(f"Extracted symptoms: {extracted_symptoms}")
        
        symptoms_text = " ".join(extracted_symptoms)
        
        try:
            diagnosis = ai_symptom_checker.predict_condition(symptoms_text)
            logger.info(f"AI diagnosis: {diagnosis}")
        except Exception as ai_error:
            logger.error(f"AI prediction failed: {ai_error}")
            diagnosis = "Unable to determine condition. Please consult a healthcare professional."
        
        symptoms_for_db = ",".join(extracted_symptoms)
        diagnosis_record = crud.add_diagnosis(
            db, 
            symptom_input.patient_id, 
            symptoms_for_db, 
            diagnosis
        )
        
        if not diagnosis_record:
            raise HTTPException(status_code=500, detail="Failed to save diagnosis")
        
        return DiagnosisResponse(
            diagnosis=diagnosis,
            extracted_symptoms=extracted_symptoms,
            user_input=symptom_input.user_input,
            patient_id=symptom_input.patient_id,
            diagnosis_id=diagnosis_record.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diagnosis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

# ------------------ Electronic Health Record ------------------
@app.get("/ehr")
def get_patient_ehr(patient_id: int, db: Session = Depends(get_db)):
    """Retrieve complete Electronic Health Record for a patient"""
    try:
        ehr = crud.get_patient_ehr(db, patient_id)
        if not ehr:
            raise HTTPException(status_code=404, detail="No EHR found for this patient")
        
        ehr["summary"] = {
            "total_vitals_records": len(ehr["vitals"]),
            "total_diagnoses": len(ehr["diagnoses"]),
            "last_visit": None
        }
        
        if ehr["vitals"] or ehr["diagnoses"]:
            all_timestamps = []
            for vital in ehr["vitals"]:
                if vital["timestamp"]:
                    all_timestamps.append(vital["timestamp"])
            for diagnosis in ehr["diagnoses"]:
                if diagnosis["timestamp"]:
                    all_timestamps.append(diagnosis["timestamp"])
            
            if all_timestamps:
                ehr["summary"]["last_visit"] = max(all_timestamps)
        
        return ehr
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EHR retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EHR retrieval failed: {str(e)}")

# ------------------ Database Testing ------------------
@app.get("/test-db")
def test_database_connection(db: Session = Depends(get_db)):
    """Test database connectivity and show statistics"""
    try:
        patient_count = db.query(models.Patient).count()
        vitals_count = db.query(models.Vitals).count()
        diagnosis_count = db.query(models.Diagnosis).count()
        
        recent_patients = db.query(models.Patient).limit(5).all()
        recent_diagnoses = db.query(models.Diagnosis).order_by(models.Diagnosis.timestamp.desc()).limit(5).all()
        
        return {
            "status": "Database connected successfully",
            "database_path": db_path,
            "statistics": {
                "total_patients": patient_count,
                "total_vitals_records": vitals_count,
                "total_diagnoses": diagnosis_count
            },
            "recent_activity": {
                "recent_patients": [{"id": p.id, "name": p.name} for p in recent_patients],
                "recent_diagnoses": [{"id": d.id, "result": d.result} for d in recent_diagnoses]
            },
            "tables_status": "All tables operational"
        }
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database test failed: {str(e)}")

# ------------------ Health Check ------------------
@app.get("/health")
def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "service": "Healthcare Kiosk API",
        "version": "1.0.0"
    }

# ------------------ Startup Event ------------------
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    print("🏥 Healthcare Kiosk API Starting...")
    print("✅ Database initialized")
    print("✅ AI Symptom Checker loaded")
    print("✅ All endpoints registered")
    print("🚀 Server ready to accept requests!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
