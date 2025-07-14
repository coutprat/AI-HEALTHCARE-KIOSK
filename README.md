# 🏥 AI-Based Healthcare Kiosk

An intelligent healthcare kiosk system that combines artificial intelligence, facial recognition, and modern web technologies to transform patient care delivery through touchless interaction and automated health screening.

![Healthcare Kiosk](https://img.shields.io/badge/AI-Healthcare-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red) ![Python](https://img.shields.io/badge/Python-3.10+-blue)

---

## 🌟 Features

### 🔐 Patient Management
- **Secure Registration**: Aadhaar-based patient registration with data validation
- **Biometric Authentication**: Automatic face recognition for touchless check-in
- **Electronic Health Records**: Comprehensive patient data management

### 🤖 AI-Powered Diagnostics
- **Symptom Analysis**: Natural language processing for symptom extraction
- **Disease Prediction**: AI-based preliminary health assessments
- **Voice Interaction**: Speech-to-text and text-to-speech capabilities

### 💓 Health Monitoring
- **Vital Signs Tracking**: Height, weight, blood pressure, pulse monitoring
- **BMI Calculation**: Automatic health category assessment
- **Health History**: Complete medical record tracking

### 🎥 Advanced Technology
- **Computer Vision**: Real-time face detection and recognition
- **Touchless Interface**: Hygienic, contactless patient interaction
- **Multi-modal Input**: Text, voice, and visual input methods

┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Frontend │<──>│ Backend │<──>│ Database │
│ (Streamlit) │ │ (FastAPI) │ │ (SQLite) │
│ Port: 8501 │ │ Port: 8000 │ │ healthcare.db │
└───────────────┘ └───────────────┘ └───────────────┘


### Frontend Components
- Streamlit Web Interface
- Face Recognition (OpenCV + face_recognition)
- Voice Processing (speech_recognition + gTTS)

### Backend Components
- FastAPI REST API
- AI Symptom Checker (NLP & ML)
- SQLAlchemy ORM with SQLite

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam for face recognition
- Microphone for voice input (optional)

### Installation

# 1. Clone the Repository
git clone https://github.com/coutprat/AI-HEALTHCARE-KIOSK.git
cd healthcare-kiosk

# 2. Create Virtual Environment (OpenVINO optimized)
python -m venv openvino_env
openvino_env\Scripts\activate  # Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Create Required Directories
mkdir frontend/encodings
mkdir known_faces
echo "" > backend/__init__.py
echo "" > backend/database/__init__.py

Note: This project uses a virtual environment named `openvino_env` to support OpenVINO-accelerated AI processing for optimal performance.

Running the Application
bash
Copy
Edit
# Terminal 1: Start Backend
cd healthcare-kiosk
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd healthcare-kiosk/frontend
streamlit run app.py --server.port 8501
Frontend: http://localhost:8501

API Docs: http://localhost:8000/docs

##📁 Project Structure


healthcare_kiosk/
├── backend/
│   ├── main.py
│   ├── ai_symptom_checker.py
│   ├── symptoms_extractor.py
│   ├── symptoms_disease.csv
│   └── database/
│       ├── __init__.py
│       ├── models.py
│       ├── crud.py
│       ├── database.py
│       └── healthcare.db
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── face_register.py
│   ├── face_checkin.py
│   └── encodings/
├── known_faces/
├── requirements.txt
└── README.md
