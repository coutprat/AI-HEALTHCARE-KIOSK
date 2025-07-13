import requests
import streamlit as st
from typing import Dict, Any, Optional
import json

class HealthcareAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with detailed error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Log the request for debugging
            if 'json' in kwargs:
                print(f"[DEBUG] {method} {url}")
                print(f"[DEBUG] Request data: {kwargs['json']}")
            
            response = self.session.request(method, url, **kwargs)
            
            # Log the response for debugging
            print(f"[DEBUG] Response status: {response.status_code}")
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                    print(f"[DEBUG] Error response: {error_detail}")
                    return {"error": error_detail.get("detail", "Unknown error"), "status_code": response.status_code}
                except:
                    print(f"[DEBUG] Raw error response: {response.text}")
                    return {"error": f"HTTP {response.status_code}: {response.text}", "status_code": response.status_code}
            
            return response.json()
            
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to backend API. Please ensure the FastAPI server is running."
            st.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.Timeout:
            error_msg = "Request timeout. Please try again."
            st.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            st.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            st.error(error_msg)
            return {"error": error_msg}
    
    def register_patient(self, aadhaar: str, name: str, age: int, gender: str) -> Dict[str, Any]:
        """Register a new patient with validation"""
        
        # Validate data before sending
        if not aadhaar or len(aadhaar) != 12:
            return {"error": "Aadhaar must be exactly 12 digits"}
        
        if not aadhaar.isdigit():
            return {"error": "Aadhaar must contain only digits"}
        
        if not name or not name.strip():
            return {"error": "Name cannot be empty"}
        
        if age < 1 or age > 120:
            return {"error": "Age must be between 1 and 120"}
        
        if gender not in ['Male', 'Female', 'Other']:
            return {"error": "Gender must be Male, Female, or Other"}
        
        data = {
            "aadhaar": aadhaar.strip(),
            "name": name.strip(),
            "age": int(age),
            "gender": gender
        }
        
        return self._make_request("POST", "/register", json=data)
    
    def login_patient(self, aadhaar: str) -> Dict[str, Any]:
        """Login patient by Aadhaar"""
        return self._make_request("GET", "/login", params={"aadhaar": aadhaar})
    
    def login_by_name(self, name: str) -> Dict[str, Any]:
        """Login patient by name (for face recognition)"""
        if not name or not name.strip():
            return {"error": "Name cannot be empty"}
        
        return self._make_request("GET", "/login-by-name", params={"name": name.strip()})
    
    def add_vitals(self, patient_id: int, height: float, weight: float, 
                   bp: str, pulse: int) -> Dict[str, Any]:
        """Add patient vitals"""
        data = {
            "patient_id": patient_id,
            "height": height,
            "weight": weight,
            "bp": bp,
            "pulse": pulse
        }
        return self._make_request("POST", "/vitals", json=data)
    
    def diagnose_symptoms(self, patient_id: int, user_input: str) -> Dict[str, Any]:
        """Get AI diagnosis for symptoms"""
        data = {
            "patient_id": patient_id,
            "user_input": user_input
        }
        return self._make_request("POST", "/diagnose", json=data)
    
    def get_patient_ehr(self, patient_id: int) -> Dict[str, Any]:
        """Get patient's Electronic Health Record"""
        return self._make_request("GET", "/ehr", params={"patient_id": patient_id})
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection"""
        return self._make_request("GET", "/")

# Global API client instance
api_client = HealthcareAPIClient()
