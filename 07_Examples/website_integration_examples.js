
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
