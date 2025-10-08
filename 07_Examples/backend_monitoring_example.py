
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
