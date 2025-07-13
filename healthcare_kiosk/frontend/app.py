import streamlit as st
import cv2
import numpy as np
from PIL import Image
import speech_recognition as sr
import pyttsx3
import tempfile
import os
from api_client import api_client
from face_register import register_face
import face_recognition
import pickle

# Page configuration
st.set_page_config(
    page_title="AI Healthcare Kiosk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def test_api_connection():
    """Test backend API connection"""
    try:
        response = api_client.test_connection()
        if "error" not in response:
            st.success("✅ Backend API Connected Successfully!")
            return True
        else:
            st.error(f"❌ Backend API Error: {response['error']}")
            return False
    except Exception as e:
        st.error(f"❌ Cannot connect to backend API: {str(e)}")
        return False

def patient_registration():
    """Patient Registration Page with improved validation"""
    st.header("🆕 Patient Registration")
    
    # Initialize session state
    if 'registration_complete' not in st.session_state:
        st.session_state.registration_complete = False
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    
    if not st.session_state.registration_complete:
        with st.form("registration_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                aadhaar = st.text_input(
                    "Aadhaar Number", 
                    max_chars=12,
                    help="Enter exactly 12 digits"
                )
                name = st.text_input(
                    "Full Name",
                    help="Enter your complete name"
                )
            
            with col2:
                age = st.number_input(
                    "Age", 
                    min_value=1, 
                    max_value=120, 
                    value=25,
                    help="Age between 1 and 120"
                )
                gender = st.selectbox(
                    "Gender", 
                    ["Male", "Female", "Other"],
                    help="Select your gender"
                )
            
            register_btn = st.form_submit_button("Register Patient")
            
            if register_btn:
                # Client-side validation
                errors = []
                
                if not aadhaar:
                    errors.append("Aadhaar number is required")
                elif len(aadhaar) != 12:
                    errors.append("Aadhaar must be exactly 12 digits")
                elif not aadhaar.isdigit():
                    errors.append("Aadhaar must contain only digits")
                
                if not name or not name.strip():
                    errors.append("Name is required")
                
                if age < 1 or age > 120:
                    errors.append("Age must be between 1 and 120")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    # Show loading spinner
                    with st.spinner("Registering patient..."):
                        response = api_client.register_patient(aadhaar, name, age, gender)
                    
                    if "error" not in response:
                        st.success(f"✅ Patient registered successfully! ID: {response['patient_id']}")
                        st.session_state.patient_data = {
                            'name': name,
                            'patient_id': response['patient_id']
                        }
                        st.session_state.registration_complete = True
                        st.rerun()
                    else:
                        st.error(f"❌ Registration failed: {response['error']}")
                        if 'status_code' in response:
                            st.error(f"Status Code: {response['status_code']}")
    
    # Step 2: Face Registration (Outside the form)
    else:
        st.success(f"✅ Patient {st.session_state.patient_data['name']} registered successfully!")
        
        st.info("📸 Now let's register your face for future check-ins...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎥 Start Face Registration", type="primary"):
                with st.spinner("Starting face registration..."):
                    success = register_face(st.session_state.patient_data['name'])
                    if success:
                        st.success("✅ Face registration completed!")
                        st.balloons()
                    else:
                        st.error("❌ Face registration failed")
        
        with col2:
            if st.button("⏭️ Skip Face Registration"):
                st.info("Face registration skipped. You can register your face later.")
                st.session_state.registration_complete = False
                st.session_state.patient_data = None
                st.rerun()

def patient_checkin():
    """Patient Check-in Page with Fixed Automatic Face Recognition"""
    st.header("🔐 Patient Check-in")
    
    tab1, tab2 = st.tabs(["Automatic Face Recognition", "Manual Aadhaar Login"])
    
    with tab1:
        st.subheader("🎥 Automatic Face Recognition Check-in")
        st.info("Look at the camera for automatic recognition")
        
        # Show current login status
        if st.session_state.patient_id:
            st.success(f"✅ Already logged in as: {st.session_state.patient_name}")
            if st.button("Continue to Symptom Checker"):
                st.session_state.current_page = "Symptom Checker"
                st.rerun()
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Automatic Recognition", type="primary", use_container_width=True):
                with st.spinner("🔍 Looking for registered faces... Please look at the camera"):
                    try:
                        # Import the automatic recognition function
                        from face_checkin import automatic_face_checkin
                        
                        # Start automatic recognition
                        recognized_user = automatic_face_checkin(timeout=20)
                        
                        if recognized_user:
                            st.success(f"✅ Face recognized: {recognized_user}")
                            
                            # Get patient details from backend using the recognized name
                            with st.spinner("Fetching patient details..."):
                                response = api_client.login_by_name(recognized_user)
                            
                            if "error" not in response:
                                # Successfully logged in
                                st.session_state.patient_id = response['patient_id']
                                st.session_state.patient_name = response['name']
                                
                                st.success(f"🎉 Welcome back, {response['name']}!")
                                st.balloons()
                                
                                # Auto-redirect to symptom checker
                                import time
                                time.sleep(2)
                                st.session_state.current_page = "Symptom Checker"
                                st.rerun()
                            else:
                                st.error(f"❌ Login failed: {response.get('error', 'Patient not found in database')}")
                                st.info("💡 The face was recognized but no matching patient record was found. Please register first.")
                        else:
                            st.error("❌ Face not recognized. Please try again or use manual login.")
                            
                    except ImportError:
                        st.error("❌ Face recognition module not available. Please use manual login.")
                    except Exception as e:
                        st.error(f"❌ Recognition failed: {str(e)}")
        
        with col2:
            st.info("**Instructions:**")
            st.write("• Position yourself in front of the camera")
            st.write("• Look directly at the camera")
            st.write("• Keep your face well-lit")
            st.write("• Wait for automatic recognition")
            st.write("• Press 'q' to cancel if needed")
    
    with tab2:
        st.subheader("📝 Manual Aadhaar Login")
        
        with st.form("login_form"):
            aadhaar = st.text_input("Enter Aadhaar Number", max_chars=12)
            login_btn = st.form_submit_button("Login")
            
            if login_btn and len(aadhaar) == 12:
                with st.spinner("Logging in..."):
                    response = api_client.login_patient(aadhaar)
                
                if "error" not in response:
                    st.success(f"✅ Welcome, {response['name']}!")
                    st.session_state.patient_id = response['patient_id']
                    st.session_state.patient_name = response['name']
                    st.session_state.current_page = "Symptom Checker"
                    st.rerun()
                else:
                    st.error("❌ Patient not found. Please register first.")

def symptom_checker():
    """AI Symptom Checker Page"""
    if not st.session_state.patient_id:
        st.error("❌ Please login first")
        return
    
    st.header("🩺 AI Symptom Checker")
    st.subheader(f"Patient: {st.session_state.patient_name}")
    
    tab1, tab2 = st.tabs(["Text Input", "Voice Input"])
    
    with tab1:
        st.subheader("Describe Your Symptoms")
        
        symptoms_text = st.text_area(
            "Enter your symptoms:",
            placeholder="e.g., I have fever, headache, and body pain",
            height=100
        )
        
        if st.button("Get Diagnosis", key="text_diagnosis"):
            if symptoms_text:
                with st.spinner("🤖 AI is analyzing your symptoms..."):
                    response = api_client.diagnose_symptoms(
                        st.session_state.patient_id, 
                        symptoms_text
                    )
                    
                    if "error" not in response:
                        st.success("✅ Diagnosis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🔍 Extracted Symptoms")
                            for symptom in response['extracted_symptoms']:
                                st.write(f"• {symptom}")
                        
                        with col2:
                            st.subheader("🏥 Possible Condition")
                            st.write(f"**{response['diagnosis']}**")
                            
                            st.info("⚠️ This is an AI prediction. Please consult a doctor for proper diagnosis.")
                    else:
                        st.error(f"❌ Diagnosis failed: {response.get('error', 'Unknown error')}")
            else:
                st.error("❌ Please enter your symptoms")
    
    with tab2:
        st.subheader("Voice Input")
        
        if st.button("🎤 Start Voice Recording"):
            try:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("🎤 Listening... Please speak your symptoms")
                    audio = r.listen(source, timeout=10)
                    
                    st.info("🔄 Processing speech...")
                    symptoms_text = r.recognize_google(audio)
                    
                    st.success(f"✅ Heard: {symptoms_text}")
                    
                    # Process with AI
                    response = api_client.diagnose_symptoms(
                        st.session_state.patient_id, 
                        symptoms_text
                    )
                    
                    if "error" not in response:
                        st.subheader("🏥 AI Diagnosis")
                        st.write(f"**Condition:** {response['diagnosis']}")
                        st.write(f"**Symptoms:** {', '.join(response['extracted_symptoms'])}")
                    
            except sr.UnknownValueError:
                st.error("❌ Could not understand audio")
            except sr.RequestError as e:
                st.error(f"❌ Speech recognition error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

def vitals_monitoring():
    """Vitals Monitoring Page"""
    if not st.session_state.patient_id:
        st.error("❌ Please login first")
        return
    
    st.header("💓 Vitals Monitoring")
    st.subheader(f"Patient: {st.session_state.patient_name}")
    
    with st.form("vitals_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
        
        with col2:
            bp = st.text_input("Blood Pressure", value="120/80")
            pulse = st.number_input("Pulse Rate", min_value=40, max_value=200, value=72)
        
        submit_vitals = st.form_submit_button("Submit Vitals")
        
        if submit_vitals:
            response = api_client.add_vitals(
                st.session_state.patient_id,
                height, weight, bp, pulse
            )
            
            if "error" not in response:
                st.success("✅ Vitals recorded successfully!")
                
                # Display BMI
                st.metric("BMI", f"{response['bmi']:.1f}")
                
                # BMI interpretation
                bmi_category = response.get('bmi_category', 'Normal')
                if bmi_category == "Underweight":
                    st.info("📊 BMI Category: Underweight")
                elif bmi_category == "Normal":
                    st.success("📊 BMI Category: Normal weight")
                elif bmi_category == "Overweight":
                    st.warning("📊 BMI Category: Overweight")
                else:
                    st.error("📊 BMI Category: Obese")
            else:
                st.error(f"❌ Failed to record vitals: {response.get('error', 'Unknown error')}")

def ehr_viewer():
    """Electronic Health Record Viewer"""
    if not st.session_state.patient_id:
        st.error("❌ Please login first")
        return
    
    st.header("📋 Electronic Health Record")
    st.subheader(f"Patient: {st.session_state.patient_name}")
    
    if st.button("🔄 Refresh EHR"):
        response = api_client.get_patient_ehr(st.session_state.patient_id)
        
        if "error" not in response:
            # Patient Information
            st.subheader("👤 Patient Information")
            patient_info = response['patient']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Name", patient_info['name'])
            with col2:
                st.metric("Age", patient_info['age'])
            with col3:
                st.metric("Gender", patient_info['gender'])
            
            # Vitals History
            st.subheader("💓 Vitals History")
            if response['vitals']:
                vitals_data = []
                for vital in response['vitals']:
                    vitals_data.append({
                        "Date": vital['timestamp'][:10] if vital['timestamp'] else "N/A",
                        "Height (cm)": vital['height'],
                        "Weight (kg)": vital['weight'],
                        "BP": vital['bp'],
                        "Pulse": vital['pulse'],
                        "BMI": vital['bmi']
                    })
                
                st.dataframe(vitals_data)
            else:
                st.info("No vitals recorded yet")
            
            # Diagnosis History
            st.subheader("🩺 Diagnosis History")
            if response['diagnoses']:
                for diagnosis in response['diagnoses']:
                    with st.expander(f"Diagnosis - {diagnosis['timestamp'][:10] if diagnosis['timestamp'] else 'N/A'}"):
                        st.write(f"**Symptoms:** {diagnosis['symptoms']}")
                        st.write(f"**Diagnosis:** {diagnosis['result']}")
            else:
                st.info("No diagnoses recorded yet")
        else:
            st.error(f"❌ Failed to load EHR: {response.get('error', 'Unknown error')}")

def main():
    """Main Streamlit Application with Debug Info"""
    st.title("🏥 AI-Based Healthcare Kiosk")
    
    # Debug session state (remove in production)
    with st.sidebar.expander("🔧 Debug Info", expanded=False):
        st.write("**Session State:**")
        st.write(f"Patient ID: {st.session_state.get('patient_id', 'None')}")
        st.write(f"Patient Name: {st.session_state.get('patient_name', 'None')}")
        st.write(f"Current Page: {st.session_state.get('current_page', 'None')}")
        
        if st.button("Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown("---")
    
    # Test API connection
    if not test_api_connection():
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if st.session_state.patient_id:
        st.sidebar.success(f"Logged in as: {st.session_state.patient_name}")
        
        pages = {
            "Symptom Checker": "🩺",
            "Vitals Monitoring": "💓",
            "EHR Viewer": "📋",
            "Logout": "🚪"
        }
    else:
        pages = {
            "Home": "🏠",
            "Register": "🆕",
            "Check-in": "🔐"
        }
    
    # Page selection
    for page, icon in pages.items():
        if st.sidebar.button(f"{icon} {page}"):
            if page == "Logout":
                st.session_state.patient_id = None
                st.session_state.patient_name = None
                st.session_state.current_page = "Home"
            else:
                st.session_state.current_page = page
            st.rerun()
    
    # Display current page
    if st.session_state.current_page == "Home":
        st.header("🏠 Welcome to AI Healthcare Kiosk")
        st.markdown("""
        ### Features:
        - 🆕 **Patient Registration** with face recognition
        - 🔐 **Secure Check-in** via face or Aadhaar
        - 🩺 **AI Symptom Checker** with voice support
        - 💓 **Vitals Monitoring** and BMI calculation
        - 📋 **Electronic Health Records** management
        
        ### Getting Started:
        1. **New patients**: Click "Register" to create your profile
        2. **Existing patients**: Use "Check-in" to access your account
        """)
        
    elif st.session_state.current_page == "Register":
        patient_registration()
    elif st.session_state.current_page == "Check-in":
        patient_checkin()
    elif st.session_state.current_page == "Symptom Checker":
        symptom_checker()
    elif st.session_state.current_page == "Vitals Monitoring":
        vitals_monitoring()
    elif st.session_state.current_page == "EHR Viewer":
        ehr_viewer()

if __name__ == "__main__":
    main()
