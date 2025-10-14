"""
FlexoTwin API - Testing dan Usage Examples
Contoh penggunaan API untuk integrasi dengan backend
"""

import requests
import json
from datetime import datetime

# API Base URL
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_system_overview():
    """Test system overview endpoint"""
    print("📊 Testing System Overview...")
    response = requests.get(f"{API_BASE_URL}/system/overview")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_components_status():
    """Test components status endpoint"""
    print("🔧 Testing Components Status...")
    response = requests.get(f"{API_BASE_URL}/components/status")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    if data["success"]:
        components = data["data"]["components"]
        print(f"Found {len(components)} components:")
        for comp in components:
            print(f"  - {comp['name']}: {comp['health_score']:.1f}% ({comp['status']})")
    print("-" * 50)

def test_file_upload():
    """Test file upload endpoint with sample CSV"""
    print("📁 Testing File Upload...")
    
    # Create sample CSV content
    csv_content = """Date,PRE_FEEDER,FEEDER,PRINTING_1,PRINTING_2,PRINTING_3,PRINTING_4,SLOTTER,DOWN_STACKER,OEE
2025-09-01,85.2,78.5,92.1,81.3,67.8,75.9,89.4,82.7,0.78
2025-09-02,84.8,79.1,91.8,82.1,68.2,76.3,88.9,83.1,0.79
2025-09-03,85.5,78.8,92.3,81.7,67.5,75.6,89.7,82.4,0.77"""
    
    # Prepare file upload
    files = {
        'file': ('production_data.csv', csv_content, 'text/csv')
    }
    
    response = requests.post(f"{API_BASE_URL}/file/upload", files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_generate_predictions():
    """Test predictions generation endpoint"""
    print("🔮 Testing Predictions Generation...")
    
    # Sample input data
    input_data = {
        "production_data": {
            "date": "2025-09-30",
            "pre_feeder": 85.2,
            "feeder": 78.5,
            "printing_1": 92.1,
            "printing_2": 81.3,
            "printing_3": 67.8,
            "printing_4": 75.9,
            "slotter": 89.4,
            "down_stacker": 82.7,
            "oee": 0.78,
            "production_hours": 16,
            "total_output": 15000
        },
        "component_data": {
            "maintenance_history": "normal",
            "operating_conditions": "standard"
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predictions/generate",
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_methodology_analysis():
    """Test methodology analysis endpoint"""
    print("📈 Testing Methodology Analysis...")
    response = requests.get(f"{API_BASE_URL}/analytics/methodology")
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    if data["success"]:
        methodology = data["data"]["methodology"]
        print(f"Found {len(methodology)} methodology steps:")
        for step in methodology:
            print(f"  Step {step['step']}: {step['title']} - {step['status']}")
    print("-" * 50)

def test_fmea_analysis():
    """Test FMEA analysis endpoint"""
    print("⚠️ Testing FMEA Analysis...")
    response = requests.get(f"{API_BASE_URL}/analytics/fmea")
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    if data["success"]:
        fmea = data["data"]["fmea"]
        print(f"Found {len(fmea)} failure modes:")
        for failure in sorted(fmea, key=lambda x: x['rpn'], reverse=True)[:3]:
            print(f"  - {failure['failure_mode']}: RPN = {failure['rpn']}")
    print("-" * 50)

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting FlexoTwin API Tests...")
    print("=" * 60)
    
    try:
        test_health_check()
        test_system_overview() 
        test_components_status()
        test_file_upload()
        test_generate_predictions()
        test_methodology_analysis()
        test_fmea_analysis()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure the API server is running:")
        print("   python flexotwin_api.py")
        print("   or")
        print("   uvicorn flexotwin_api:app --host 0.0.0.0 --port 8000")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    run_all_tests()