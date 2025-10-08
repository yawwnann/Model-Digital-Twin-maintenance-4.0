"""
Simple API Test Script
Test FlexoTwin API endpoints
"""

import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing FlexoTwin API...")
    print("=" * 50)
    
    # Test 1: Health Check
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"✅ Health Check: {response.status_code}")
        if response.status_code == 200:
            print(f"   Status: {response.json().get('status')}")
    except Exception as e:
        print(f"❌ Health Check Error: {str(e)}")
    
    # Test 2: Machine Status
    try:
        response = requests.get(f"{base_url}/api/machine_status")
        print(f"✅ Machine Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            current = data.get('current_status', {})
            print(f"   OEE: {current.get('oee', 0):.1%}")
            print(f"   Downtime: {current.get('downtime_minutes', 0)} minutes")
            print(f"   Alerts: {len(data.get('alerts', []))} active")
    except Exception as e:
        print(f"❌ Machine Status Error: {str(e)}")
    
    # Test 3: Maintenance Prediction
    machine_data = {
        "OEE": 0.15,
        "Performance": 0.2, 
        "Quality": 0.95,
        "Downtime": 180,
        "Scrab_Rate": 0.05,
        "Shift": 1,
        "Day_of_Week": 1,
        "Month_Num": 10
    }
    
    try:
        response = requests.post(f"{base_url}/api/predict_maintenance", json=machine_data)
        print(f"✅ Maintenance Prediction: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            pred = result.get('prediction', {})
            print(f"   Days until maintenance: {pred.get('days_until_maintenance')}")
            print(f"   Risk level: {pred.get('risk_level')}")
            print(f"   Maintenance type: {pred.get('maintenance_type')}")
            print(f"   Urgency score: {pred.get('urgency_score')}")
            
            actions = pred.get('recommended_actions', [])
            print(f"   Recommended actions: {len(actions)} items")
            for i, action in enumerate(actions[:3], 1):
                print(f"      {i}. {action}")
                
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
    
    # Test 4: Dashboard Data
    try:
        response = requests.get(f"{base_url}/api/dashboard_data")
        print(f"✅ Dashboard Data: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print(f"   Current OEE: {summary.get('current_oee', 0):.1%}")
            print(f"   Target OEE: {summary.get('target_oee', 0):.1%}")
            print(f"   Total records: {summary.get('total_records', 0):,}")
            
    except Exception as e:
        print(f"❌ Dashboard Error: {str(e)}")
    
    print("\n🎉 API Testing Completed!")

if __name__ == "__main__":
    test_api()