"""
FlexoTwin API Testing & Usage Examples
Contoh penggunaan API untuk integrasi dengan website
"""

import requests
import json
from datetime import datetime

class FlexoTwinAPIClient:
    def __init__(self, base_url="http://localhost:5000"):
        """Initialize API client"""
        self.base_url = base_url
        
    def test_maintenance_prediction(self):
        """
        Test maintenance prediction endpoint
        """
        print("🔧 Testing Maintenance Prediction API...")
        print("=" * 50)
        
        # Example machine data
        machine_data = {
            "OEE": 0.15,              # Low OEE - 15%
            "Availability": 0.8,      # Good availability - 80%
            "Performance": 0.2,       # Poor performance - 20%
            "Quality": 0.95,          # Good quality - 95%
            "Downtime": 180,          # 3 hours downtime
            "Scrab_Rate": 0.05,       # 5% scrap rate
            "Total_Production": 1000,
            "Shift": 1,               # Day shift
            "Day_of_Week": 1,         # Monday
            "Month_Num": 10           # October
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/predict_maintenance",
                json=machine_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API Response Success!")
                print(json.dumps(result, indent=2))
                
                # Extract key information
                prediction = result.get('prediction', {})
                print(f"\n📊 KEY RESULTS:")
                print(f"   Days Until Maintenance: {prediction.get('days_until_maintenance')} days")
                print(f"   Maintenance Date: {prediction.get('maintenance_date')}")
                print(f"   Risk Level: {prediction.get('risk_level')}")
                print(f"   Maintenance Type: {prediction.get('maintenance_type')}")
                print(f"   Estimated Cost: Rp {prediction.get('estimated_cost', {}).get('min', 0):,} - Rp {prediction.get('estimated_cost', {}).get('max', 0):,}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: API server not running")
            print("💡 Start server first: python 07_api_interface.py")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def test_machine_status(self):
        """Test machine status endpoint"""
        print("\n📊 Testing Machine Status API...")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/api/machine_status")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Machine Status Retrieved!")
                
                current_status = result.get('current_status', {})
                print(f"   OEE: {current_status.get('oee', 0):.1%}")
                print(f"   Performance: {current_status.get('performance', 0):.1%}")
                print(f"   Downtime: {current_status.get('downtime_minutes', 0)} minutes")
                
                alerts = result.get('alerts', [])
                if alerts:
                    print(f"   ⚠️  Alerts: {len(alerts)} active")
                    for alert in alerts:
                        print(f"      - {alert.get('message', 'N/A')}")
                else:
                    print(f"   ✅ No active alerts")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def test_dashboard_data(self):
        """Test dashboard data endpoint"""
        print("\n📈 Testing Dashboard Data API...")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/api/dashboard_data")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Dashboard Data Retrieved!")
                
                summary = result.get('summary', {})
                key_metrics = result.get('key_metrics', {})
                
                print(f"   Current OEE: {summary.get('current_oee', 0):.1%}")
                print(f"   Target OEE: {summary.get('target_oee', 0):.1%}")
                print(f"   Performance Status: {summary.get('oee_performance', 'N/A')}")
                print(f"   Total Downtime: {key_metrics.get('total_downtime_hours', 0)} hours")
                print(f"   Failure Rate: {key_metrics.get('failure_rate', 0):.1%}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def create_website_integration_examples():
    """
    Create example code untuk website integration
    """
    print("\n💻 Website Integration Examples")
    print("=" * 60)
    
    # JavaScript example untuk frontend
    js_example = '''
// JavaScript Example - Frontend Integration
async function getPrediction() {
    const machineData = {
        OEE: 0.15,
        Performance: 0.2,
        Quality: 0.95,
        Downtime: 180,
        Scrab_Rate: 0.05
    };
    
    try {
        const response = await fetch('http://localhost:5000/api/predict_maintenance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(machineData)
        });
        
        const result = await response.json();
        
        // Update UI dengan prediction results
        document.getElementById('maintenance-date').textContent = result.prediction.maintenance_date;
        document.getElementById('risk-level').textContent = result.prediction.risk_level;
        document.getElementById('urgency-score').textContent = result.prediction.urgency_score;
        
        // Show recommendations
        const actionsList = document.getElementById('recommended-actions');
        actionsList.innerHTML = '';
        result.prediction.recommended_actions.forEach(action => {
            const li = document.createElement('li');
            li.textContent = action;
            actionsList.appendChild(li);
        });
        
    } catch (error) {
        console.error('API Error:', error);
    }
}

// Update dashboard data
async function updateDashboard() {
    try {
        const response = await fetch('http://localhost:5000/api/dashboard_data');
        const data = await response.json();
        
        // Update OEE gauge
        updateOEEGauge(data.summary.current_oee);
        
        // Update charts
        updatePerformanceChart(data.monthly_performance);
        
    } catch (error) {
        console.error('Dashboard update error:', error);
    }
}

// Real-time updates
setInterval(updateDashboard, 30000); // Update every 30 seconds
'''
    
    # Python example untuk backend processing
    python_example = '''
# Python Example - Backend Processing
import requests
import schedule
import time

class MaintenanceMonitor:
    def __init__(self):
        self.api_url = "http://localhost:5000"
    
    def check_maintenance_status(self):
        """Check maintenance status every hour"""
        try:
            # Get current machine status
            response = requests.get(f"{self.api_url}/api/machine_status")
            status = response.json()
            
            # Check for critical alerts
            alerts = status.get('alerts', [])
            critical_alerts = [a for a in alerts if a.get('severity') == 'high']
            
            if critical_alerts:
                self.send_notification(critical_alerts)
                
        except Exception as e:
            print(f"Monitoring error: {str(e)}")
    
    def send_notification(self, alerts):
        """Send notification untuk critical alerts"""
        # Implementation: email, SMS, push notification, etc.
        for alert in alerts:
            print(f"CRITICAL ALERT: {alert.get('message')}")
    
    def start_monitoring(self):
        """Start automated monitoring"""
        schedule.every().hour.do(self.check_maintenance_status)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

# Usage
monitor = MaintenanceMonitor()
monitor.start_monitoring()
'''
    
    # Save examples ke files
    with open('website_integration_examples.js', 'w') as f:
        f.write(js_example)
    
    with open('backend_monitoring_example.py', 'w') as f:
        f.write(python_example)
    
    print("✅ Integration examples created:")
    print("   - website_integration_examples.js")
    print("   - backend_monitoring_example.py")

def create_api_documentation():
    """
    Create comprehensive API documentation
    """
    documentation = '''
# FlexoTwin Smart Maintenance API Documentation

## Overview
REST API untuk sistem prediksi maintenance mesin Flexo 104. API ini menyediakan endpoints untuk:
- Prediksi jadwal maintenance
- Monitoring status mesin real-time
- Data dashboard untuk visualisasi website
- Riwayat maintenance

## Base URL
```
http://localhost:5000
```

## Authentication
Saat ini tidak memerlukan authentication (untuk development). 
Untuk production, implementasikan JWT token authentication.

## Endpoints

### 1. Predict Maintenance
**POST** `/api/predict_maintenance`

Prediksi kapan maintenance diperlukan berdasarkan kondisi mesin saat ini.

#### Request Body
```json
{
    "OEE": 0.15,              // Overall Equipment Effectiveness (0-1)
    "Availability": 0.8,      // Equipment availability (0-1)  
    "Performance": 0.2,       // Performance score (0-1)
    "Quality": 0.95,          // Quality score (0-1)
    "Downtime": 180,          // Current downtime in minutes
    "Scrab_Rate": 0.05,       // Scrap/waste rate (0-1)
    "Total_Production": 1000, // Production quantity
    "Shift": 1,               // Shift number (1,2,3)
    "Day_of_Week": 1,         // Day of week (0=Sunday)
    "Month_Num": 10           // Month number (1-12)
}
```

#### Response
```json
{
    "status": "success",
    "prediction": {
        "days_until_maintenance": 6.1,
        "maintenance_date": "2025-10-15",
        "risk_level": "Critical",
        "urgency_score": 85.7,
        "maintenance_type": "Emergency",
        "priority": "CRITICAL - Immediate Action Required",
        "estimated_cost": {
            "min": 50000000,
            "max": 100000000,
            "currency": "IDR",
            "category": "High"
        },
        "estimated_downtime_hours": "4-8",
        "recommended_actions": [
            "Stop production immediately",
            "Emergency equipment inspection",
            "Replace critical components"
        ],
        "components_to_check": [
            "Hydraulic System",
            "Print Cylinders",
            "Drive Motors"
        ]
    },
    "machine_status": {
        "oee": 0.15,
        "availability": 0.8,
        "performance": 0.2,
        "quality": 0.95,
        "current_downtime": 180,
        "scrap_rate": 0.05
    },
    "timestamp": "2025-10-09T10:30:00.000Z"
}
```

### 2. Machine Status
**GET** `/api/machine_status`

Get status mesin saat ini dan alerts aktif.

#### Response
```json
{
    "status": "success",
    "machine_id": "C_FL104",
    "machine_name": "Flexo 104",
    "current_status": {
        "oee": 0.176,
        "availability": 0.766,
        "performance": 0.230,
        "quality": 0.780,
        "downtime_minutes": 245,
        "scrap_rate": 0.220,
        "production_qty": 1500
    },
    "last_updated": "2025-10-09T10:30:00.000Z",
    "alerts": [
        {
            "type": "warning",
            "message": "OEE below target (60%)",
            "severity": "medium"
        }
    ]
}
```

### 3. Dashboard Data
**GET** `/api/dashboard_data`

Comprehensive data untuk dashboard website, termasuk summary dan trend analysis.

#### Response
```json
{
    "status": "success",
    "summary": {
        "current_oee": 0.176,
        "target_oee": 0.85,
        "oee_performance": "Critical",
        "total_records": 5579,
        "analysis_period": "Feb 2025 - Jun 2025"
    },
    "monthly_performance": {
        "Februari": {"OEE": 0.165, "Downtime": 15420, "Total_Production": 125000},
        "Maret": {"OEE": 0.172, "Downtime": 14890, "Total_Production": 128000},
        // ... more months
    },
    "key_metrics": {
        "average_oee": 0.176,
        "failure_rate": 0.814,
        "total_downtime_hours": 22822,
        "average_downtime_per_record": 245
    },
    "recommendations": [
        "Focus on Performance improvement",
        "Implement predictive maintenance program",
        "Reduce chronic downtime issues"
    ],
    "last_updated": "2025-10-09T10:30:00.000Z"
}
```

### 4. Maintenance History  
**GET** `/api/maintenance_history`

Riwayat maintenance yang telah dilakukan.

### 5. Health Check
**GET** `/api/health`

Check API server status.

## Error Responses
```json
{
    "status": "error",
    "message": "Error description",
    "timestamp": "2025-10-09T10:30:00.000Z"
}
```

## Status Codes
- `200` - Success
- `400` - Bad Request (invalid input)
- `500` - Internal Server Error

## Integration Guidelines

### Frontend Integration
1. **Real-time Updates**: Poll `/api/machine_status` every 30 seconds
2. **Dashboard**: Load `/api/dashboard_data` on page load dan refresh every 5 minutes  
3. **Predictions**: Call `/api/predict_maintenance` when user inputs new data
4. **Error Handling**: Implement proper error handling untuk network issues

### Backend Integration
1. **Automated Monitoring**: Schedule periodic checks untuk critical alerts
2. **Data Logging**: Log all API responses untuk audit trail
3. **Notifications**: Implement alert system untuk critical maintenance needs
4. **Database**: Store prediction results untuk historical analysis

### Performance Considerations
- Implement caching untuk dashboard data
- Use WebSockets untuk real-time updates (future enhancement)
- Optimize database queries untuk large datasets
- Consider API rate limiting untuk production

### Security (Production)
- Implement JWT authentication
- Use HTTPS only
- Validate all input data
- Implement API rate limiting
- Add logging dan monitoring

## Testing
Run test script:
```bash
python 08_api_testing.py
```

## Starting the API Server
```bash
python 07_api_interface.py
```

API akan running di `http://localhost:5000`
'''
    
    with open('API_DOCUMENTATION.md', 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print("✅ API Documentation created: API_DOCUMENTATION.md")

def main():
    """Main testing function"""
    print("🧪 FlexoTwin API Testing & Examples")
    print("=" * 70)
    
    # Create integration examples
    create_website_integration_examples()
    
    # Create API documentation
    create_api_documentation()
    
    # Initialize API client
    client = FlexoTwinAPIClient()
    
    print("\n🔧 API Testing (requires API server running)")
    print("💡 Start API server first: python 05_API/07_api_interface.py")
    print("=" * 70)
    
    # Test endpoints
    client.test_maintenance_prediction()
    client.test_machine_status() 
    client.test_dashboard_data()
    
    print("\n✅ Testing completed!")
    print("\n📋 Files created untuk website integration:")
    print("   1. website_integration_examples.js - Frontend JavaScript")
    print("   2. backend_monitoring_example.py - Backend Python")
    print("   3. API_DOCUMENTATION.md - Complete API docs")
    print("\n🚀 Ready untuk website integration!")

if __name__ == "__main__":
    main()