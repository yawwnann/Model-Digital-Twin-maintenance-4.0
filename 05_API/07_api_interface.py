"""
FlexoTwin API Interface untuk Website Integration
REST API untuk sistem maintenance prediction

Endpoints:
1. /predict_maintenance - Prediksi maintenance schedule
2. /get_machine_status - Status mesin real-time
3. /get_maintenance_history - History maintenance
4. /update_machine_data - Update data mesin
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)  # Enable CORS untuk website integration

class FlexoTwinAPI:
    def __init__(self):
        """Initialize FlexoTwin API"""
        self.maintenance_model = None
        self.risk_model = None
        self.feature_names = None
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            # Update paths to new model folder structure
            self.maintenance_model = joblib.load('../02_Models/maintenance_days_predictor.joblib')
            self.risk_model = joblib.load('../02_Models/maintenance_risk_classifier.joblib')
            self.feature_names = joblib.load('../02_Models/maintenance_feature_names.joblib')
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            # Fallback: try loading from current directory (for backwards compatibility)
            try:
                self.maintenance_model = joblib.load('maintenance_days_predictor.joblib')
                self.risk_model = joblib.load('maintenance_risk_classifier.joblib')
                self.feature_names = joblib.load('maintenance_feature_names.joblib')
                print("✅ Models loaded from current directory")
            except Exception as e2:
                print(f"❌ Fallback loading failed: {str(e2)}")
    
    def predict_maintenance(self, machine_data):
        """
        Prediksi maintenance berdasarkan data mesin
        
        Args:
            machine_data (dict): Data kondisi mesin saat ini
            
        Returns:
            dict: Hasil prediksi maintenance
        """
        try:
            # Prepare input data
            input_df = pd.DataFrame([machine_data])
            
            # Ensure all features ada
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[self.feature_names]
            
            # Predictions
            days_until_maintenance = self.maintenance_model.predict(input_df)[0]
            risk_level = self.risk_model.predict(input_df)[0]
            
            # Calculate additional metrics
            urgency_score = self._calculate_urgency_score(machine_data)
            maintenance_cost = self._estimate_maintenance_cost(urgency_score, days_until_maintenance)
            
            # Generate schedule
            today = datetime.now()
            maintenance_date = today + timedelta(days=max(1, int(days_until_maintenance)))
            
            # Determine maintenance type dan priority
            maintenance_info = self._get_maintenance_type_info(urgency_score, days_until_maintenance)
            
            return {
                "status": "success",
                "prediction": {
                    "days_until_maintenance": round(float(days_until_maintenance), 1),
                    "maintenance_date": maintenance_date.strftime("%Y-%m-%d"),
                    "risk_level": self._get_risk_name(risk_level),
                    "urgency_score": round(urgency_score, 1),
                    "maintenance_type": maintenance_info["type"],
                    "priority": maintenance_info["priority"],
                    "estimated_cost": maintenance_cost,
                    "estimated_downtime_hours": maintenance_info["downtime_hours"],
                    "recommended_actions": maintenance_info["actions"],
                    "components_to_check": maintenance_info["components"]
                },
                "machine_status": {
                    "oee": machine_data.get('OEE', 0),
                    "availability": machine_data.get('Availability', 0),
                    "performance": machine_data.get('Performance', 0),
                    "quality": machine_data.get('Quality', 0),
                    "current_downtime": machine_data.get('Downtime', 0),
                    "scrap_rate": machine_data.get('Scrab_Rate', 0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Prediction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_urgency_score(self, data):
        """Calculate maintenance urgency score"""
        oee = data.get('OEE', 0.5)
        downtime = data.get('Downtime', 0)
        scrap_rate = data.get('Scrab_Rate', 0)
        performance = data.get('Performance', 0.5)
        
        # Normalize downtime (assume max 480 minutes = 8 hours)
        normalized_downtime = min(downtime / 480, 1)
        
        urgency = (
            (1 - oee) * 30 +
            normalized_downtime * 25 +
            scrap_rate * 20 +
            (1 - performance) * 25
        )
        
        return min(urgency * 100, 100)
    
    def _estimate_maintenance_cost(self, urgency_score, days_until):
        """Estimate maintenance cost dalam Rupiah"""
        if urgency_score >= 80 or days_until <= 2:
            return {"min": 50000000, "max": 100000000, "currency": "IDR", "category": "High"}
        elif urgency_score >= 60 or days_until <= 5:
            return {"min": 20000000, "max": 50000000, "currency": "IDR", "category": "Medium"}
        elif urgency_score >= 40 or days_until <= 10:
            return {"min": 5000000, "max": 20000000, "currency": "IDR", "category": "Low"}
        else:
            return {"min": 1000000, "max": 5000000, "currency": "IDR", "category": "Very Low"}
    
    def _get_maintenance_type_info(self, urgency_score, days_until):
        """Get detailed maintenance type information"""
        if urgency_score >= 80 or days_until <= 2:
            return {
                "type": "Emergency",
                "priority": "Critical",
                "downtime_hours": "4-8",
                "actions": [
                    "Stop production immediately",
                    "Emergency equipment inspection",
                    "Replace critical components",
                    "Full system check before restart"
                ],
                "components": ["Hydraulic System", "Print Cylinders", "Drive Motors", "Control Systems"]
            }
        elif urgency_score >= 60 or days_until <= 5:
            return {
                "type": "Preventive",
                "priority": "High",
                "downtime_hours": "2-4", 
                "actions": [
                    "Schedule within 48 hours",
                    "Prepare replacement parts",
                    "Inspect wear components",
                    "Check fluid systems"
                ],
                "components": ["Filters", "Belts", "Bearings", "Fluid Systems"]
            }
        elif urgency_score >= 40 or days_until <= 10:
            return {
                "type": "Scheduled",
                "priority": "Medium",
                "downtime_hours": "1-2",
                "actions": [
                    "Plan maintenance next week",
                    "Order standard parts",
                    "Review maintenance history",
                    "Schedule during low production"
                ],
                "components": ["Routine Inspection", "Lubrication", "Calibration"]
            }
        else:
            return {
                "type": "Routine",
                "priority": "Low",
                "downtime_hours": "0.5-1",
                "actions": [
                    "Continue regular schedule",
                    "Monitor performance",
                    "Update documentation",
                    "Basic maintenance checks"
                ],
                "components": ["General Inspection", "Cleaning", "Documentation"]
            }
    
    def _get_risk_name(self, risk_code):
        """Convert risk code to name"""
        risk_names = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        return risk_names.get(risk_code, "Unknown")

# Initialize API
flexo_api = FlexoTwinAPI()

@app.route('/api/predict_maintenance', methods=['POST'])
def predict_maintenance():
    """
    API Endpoint: Prediksi Maintenance
    
    Expected JSON input:
    {
        "OEE": 0.15,
        "Availability": 0.8,
        "Performance": 0.2,
        "Quality": 0.95,
        "Downtime": 180,
        "Scrab_Rate": 0.05,
        "Total_Production": 1000,
        "Shift": 1,
        "Day_of_Week": 1,
        "Month_Num": 10
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Validate required fields
        required_fields = ['OEE', 'Performance', 'Quality']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {missing_fields}"
            }), 400
        
        # Get prediction
        result = flexo_api.predict_maintenance(data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/machine_status', methods=['GET'])
def get_machine_status():
    """
    API Endpoint: Get current machine status
    """
    try:
        # Load latest data (in real app, this would be from database)
        df = pd.read_csv('flexotwin_processed_data.csv')
        latest = df.iloc[-1]
        
        status = {
            "status": "success",
            "machine_id": "C_FL104",
            "machine_name": "Flexo 104",
            "current_status": {
                "oee": round(float(latest.get('OEE', 0)), 3),
                "availability": round(float(latest.get('Availability', 0)), 3),
                "performance": round(float(latest.get('Performance', 0)), 3),
                "quality": round(float(latest.get('Quality', 0)), 3),
                "downtime_minutes": int(latest.get('Downtime', 0)),
                "scrap_rate": round(float(latest.get('Scrab_Rate', 0)), 3),
                "production_qty": int(latest.get('Total_Production', 0))
            },
            "last_updated": datetime.now().isoformat(),
            "alerts": []
        }
        
        # Add alerts berdasarkan kondisi
        if status["current_status"]["oee"] < 0.6:
            status["alerts"].append({
                "type": "warning",
                "message": "OEE below target (60%)",
                "severity": "medium"
            })
        
        if status["current_status"]["downtime_minutes"] > 240:  # > 4 hours
            status["alerts"].append({
                "type": "alert",
                "message": "High downtime detected",
                "severity": "high"
            })
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/maintenance_history', methods=['GET'])
def get_maintenance_history():
    """
    API Endpoint: Get maintenance prediction history
    """
    try:
        # Load maintenance data
        df = pd.read_csv('flexotwin_maintenance_data.csv')
        
        # Process maintenance history (example)
        history = []
        
        # Create sample maintenance records (in real app, from database)
        for i in range(min(10, len(df))):
            record = {
                "date": (datetime.now() - timedelta(days=i*7)).strftime("%Y-%m-%d"),
                "type": ["Routine", "Preventive", "Emergency"][i % 3],
                "duration_hours": np.random.uniform(1, 6),
                "cost_idr": np.random.uniform(5000000, 50000000),
                "components": ["Hydraulic System", "Print Quality", "Mechanical"][i % 3],
                "status": "Completed"
            }
            history.append(record)
        
        return jsonify({
            "status": "success",
            "maintenance_history": history,
            "total_records": len(history)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    """
    API Endpoint: Get comprehensive dashboard data untuk website
    """
    try:
        # Load processed data
        df = pd.read_csv('flexotwin_processed_data.csv')
        
        # Calculate summary statistics
        current_oee = df['OEE'].iloc[-10:].mean()  # Last 10 records average
        monthly_stats = df.groupby('Month').agg({
            'OEE': 'mean',
            'Downtime': 'sum',
            'Total_Production': 'sum',
            'Equipment_Failure': 'mean'
        }).round(3)
        
        dashboard_data = {
            "status": "success",
            "summary": {
                "current_oee": round(current_oee, 3),
                "target_oee": 0.85,
                "oee_performance": "Critical" if current_oee < 0.6 else "Warning" if current_oee < 0.75 else "Good",
                "total_records": len(df),
                "analysis_period": "Feb 2025 - Jun 2025"
            },
            "monthly_performance": monthly_stats.to_dict('index'),
            "key_metrics": {
                "average_oee": round(df['OEE'].mean(), 3),
                "failure_rate": round(df['Equipment_Failure'].mean(), 3),
                "total_downtime_hours": round(df['Downtime'].sum() / 60, 1),
                "average_downtime_per_record": round(df['Downtime'].mean(), 1)
            },
            "recommendations": [
                "Focus on Performance improvement (lowest OEE component)",
                "Implement predictive maintenance program", 
                "Reduce chronic downtime issues",
                "Enhance operator training"
            ],
            "last_updated": datetime.now().isoformat()
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API Health check"""
    return jsonify({
        "status": "healthy",
        "service": "FlexoTwin API",
        "version": "1.0",
        "timestamp": datetime.now().isoformat()
    })

# API Documentation endpoint
@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API Documentation"""
    docs = {
        "FlexoTwin Smart Maintenance API": {
            "version": "1.0",
            "description": "REST API untuk sistem prediksi maintenance mesin Flexo 104",
            "endpoints": {
                "/api/predict_maintenance": {
                    "method": "POST",
                    "description": "Prediksi jadwal maintenance berdasarkan kondisi mesin",
                    "input": {
                        "OEE": "Overall Equipment Effectiveness (0-1)",
                        "Performance": "Performance score (0-1)",
                        "Quality": "Quality score (0-1)",
                        "Downtime": "Downtime in minutes",
                        "Scrab_Rate": "Scrap rate (0-1)"
                    }
                },
                "/api/machine_status": {
                    "method": "GET", 
                    "description": "Status mesin saat ini"
                },
                "/api/maintenance_history": {
                    "method": "GET",
                    "description": "Riwayat maintenance"
                },
                "/api/dashboard_data": {
                    "method": "GET",
                    "description": "Data lengkap untuk dashboard website"
                }
            }
        }
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    print("🚀 Starting FlexoTwin API Server...")
    print("📊 Endpoints available:")
    print("   • POST /api/predict_maintenance - Maintenance prediction")
    print("   • GET /api/machine_status - Current machine status")
    print("   • GET /api/maintenance_history - Maintenance history")
    print("   • GET /api/dashboard_data - Dashboard data")
    print("   • GET /api/health - Health check")
    print("   • GET /api/docs - API documentation")
    print("\n💡 Ready for website integration!")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)