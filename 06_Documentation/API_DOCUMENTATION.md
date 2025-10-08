
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
