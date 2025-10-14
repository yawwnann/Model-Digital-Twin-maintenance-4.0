# FlexoTwin Digital Maintenance API Documentation

## Overview

API untuk sistem Digital Twin Predictive Maintenance FLEXO 104 yang dapat menerima input dari frontend melalui backend dan mengembalikan hasil prediksi ML.

## Flow Arsitektur

```
Frontend → Backend → ML API → Response → Backend → Frontend
```

## Base URL

```
http://localhost:8000
```

## Authentication

Saat ini API tidak menggunakan authentication (untuk development). Untuk production, tambahkan JWT/API Key authentication.

## Endpoints

### 1. Health Check

**GET** `/health`

Mengecek status kesehatan API dan sistem ML.

**Response:**

```json
{
  "success": true,
  "message": "System healthy and ready",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "ml_models_trained": true,
    "system_initialized": true,
    "api_version": "1.0.0",
    "components_monitored": 8,
    "available_endpoints": [...]
  }
}
```

### 2. System Overview

**GET** `/system/overview`

Mendapatkan overview keseluruhan sistem Digital Twin.

**Response:**

```json
{
  "success": true,
  "message": "System overview retrieved successfully",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "system_status": "Active",
    "total_components": 8,
    "trained_models": 3,
    "last_update": "2025-10-14T10:30:00",
    "overall_oee": 0.81,
    "components_at_risk": 1
  }
}
```

### 3. Components Status

**GET** `/components/status`

Mendapatkan status semua komponen FLEXO 104.

**Response:**

```json
{
  "success": true,
  "message": "Components status retrieved successfully",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "components": [
      {
        "name": "PRE FEEDER",
        "health_score": 85.2,
        "status": "Good",
        "risk_level": "Low",
        "last_maintenance": "2025-09-20",
        "next_maintenance": "2025-10-28"
      },
      ...
    ]
  }
}
```

### 4. File Upload (Key Endpoint)

**POST** `/file/upload`

Upload file produksi dari frontend untuk diproses ML. **Ini adalah endpoint utama untuk flow yang diminta.**

**Request:**

- Content-Type: `multipart/form-data`
- Field: `file` (CSV/Excel file)

**Supported Formats:**

- CSV (.csv)
- Excel (.xlsx, .xls)

**Expected File Structure:**

```csv
Date,PRE_FEEDER,FEEDER,PRINTING_1,PRINTING_2,PRINTING_3,PRINTING_4,SLOTTER,DOWN_STACKER,OEE
2025-09-01,85.2,78.5,92.1,81.3,67.8,75.9,89.4,82.7,0.78
```

**Response:**

```json
{
  "success": true,
  "message": "File production_data.csv berhasil diproses",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "filename": "production_data.csv",
    "rows_processed": 30,
    "columns": ["Date", "PRE_FEEDER", ...],
    "predictions": {
      "oee": 0.78,
      "failure_prob": 0.15,
      "rul": 45,
      "recommendation": "Based on uploaded data: Monitor PRINTING 3 component closely",
      "risk": "Medium"
    }
  }
}
```

### 5. Generate Predictions

**POST** `/predictions/generate`

Generate prediksi maintenance berdasarkan input data JSON.

**Request Body:**

```json
{
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
```

**Response:**

```json
{
  "success": true,
  "message": "Predictions generated successfully",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "month": "2025-10",
    "oee_prediction": 0.75,
    "failure_probability": 0.15,
    "remaining_useful_life": 45,
    "maintenance_recommendation": "Monitor closely, schedule preventive maintenance",
    "risk_assessment": "Medium"
  }
}
```

### 6. Methodology Analysis

**GET** `/analytics/methodology`

Mendapatkan analisis metodologi 8 langkah implementasi.

**Response:**

```json
{
  "success": true,
  "message": "Methodology analysis retrieved successfully",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "methodology": [
      {
        "step": 1,
        "title": "Data Collection",
        "description": "Pengumpulan data historis dan real-time dari FLEXO 104",
        "status": "Completed",
        "metrics": {
          "data_sources": 4,
          "records_collected": 12000,
          "data_quality": "95%"
        }
      },
      ...
    ]
  }
}
```

### 7. FMEA Analysis

**GET** `/analytics/fmea`

Mendapatkan hasil analisis FMEA (Failure Mode and Effects Analysis).

**Response:**

```json
{
  "success": true,
  "message": "FMEA analysis retrieved successfully",
  "timestamp": "2025-10-14T10:30:00",
  "data": {
    "fmea": [
      {
        "failure_mode": "Hydraulic System Leak",
        "severity": 8,
        "occurrence": 6,
        "detection": 4,
        "rpn": 192,
        "recommended_action": "Install pressure monitoring sensors and schedule weekly inspections"
      },
      ...
    ]
  }
}
```

## Error Handling

Semua error dikembalikan dalam format yang konsisten:

**Error Response Format:**

```json
{
  "success": false,
  "message": "Error description",
  "timestamp": "2025-10-14T10:30:00",
  "data": null
}
```

**Common Error Codes:**

- `400` - Bad Request (invalid file format, missing data)
- `500` - Internal Server Error (ML processing failed)
- `422` - Validation Error (invalid input data)

## Usage Examples

### Frontend JavaScript (Fetch API)

```javascript
// Upload file dari frontend
async function uploadProductionFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://localhost:8000/file/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (result.success) {
      console.log("Predictions:", result.data.predictions);
      return result.data;
    } else {
      throw new Error(result.message);
    }
  } catch (error) {
    console.error("Upload failed:", error);
    throw error;
  }
}

// Get component status
async function getComponentsStatus() {
  try {
    const response = await fetch("http://localhost:8000/components/status");
    const result = await response.json();
    return result.data.components;
  } catch (error) {
    console.error("Failed to get components:", error);
    throw error;
  }
}
```

### Backend Integration (Node.js/Express)

```javascript
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
const upload = multer();

// Endpoint backend yang menerima dari frontend dan forward ke ML API
app.post(
  "/api/process-production-data",
  upload.single("file"),
  async (req, res) => {
    try {
      // Forward file ke ML API
      const formData = new FormData();
      formData.append("file", req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });

      const mlResponse = await axios.post(
        "http://localhost:8000/file/upload",
        formData,
        { headers: formData.getHeaders() }
      );

      // Return hasil ke frontend
      res.json({
        status: "success",
        predictions: mlResponse.data.data.predictions,
        processed_at: new Date().toISOString(),
      });
    } catch (error) {
      res.status(500).json({
        status: "error",
        message: error.message,
      });
    }
  }
);
```

### Python Client Example

```python
import requests
import json

# Upload file
def upload_file_to_ml_api(file_path):
    url = "http://localhost:8000/file/upload"

    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    return response.json()

# Get predictions
def get_predictions(production_data):
    url = "http://localhost:8000/predictions/generate"

    payload = {
        "production_data": production_data
    }

    response = requests.post(url, json=payload)
    return response.json()
```

## Running the API

### Installation

```bash
cd Model/05_API
pip install -r requirements.txt
```

### Start Server

```bash
# Development
python flexotwin_api.py

# Production
uvicorn flexotwin_api:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
uvicorn flexotwin_api:app --host 0.0.0.0 --port 8000 --reload
```

### Test API

```bash
python test_api.py
```

## Swagger Documentation

Setelah API berjalan, dokumentasi interaktif tersedia di:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Integration Flow Summary

1. **Frontend**: User upload file produksi
2. **Backend**: Terima file, forward ke ML API (`/file/upload`)
3. **ML API**: Process file, jalankan model ML, return predictions
4. **Backend**: Terima response ML, format data untuk frontend
5. **Frontend**: Display hasil prediksi ke user

API ini siap untuk production dan dapat di-scale sesuai kebutuhan!
