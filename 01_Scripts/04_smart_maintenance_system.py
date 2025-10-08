"""
FlexoTwin Smart Maintenance 4.0
Prediction Interface & Deployment Script

Tujuan: 
1. Load trained models untuk real-time prediction
2. Create prediction interface untuk production use
3. Monitoring dashboard untuk maintenance decisions
4. Alert system untuk preventive maintenance

Dibuat untuk: Proyek Skripsi Teknik Industri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class FlexoSmartMaintenanceSystem:
    def __init__(self):
        """
        Inisialisasi FlexoTwin Smart Maintenance System
        """
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        self.feature_names = None
        self.model_summary = None
        
        # Load trained models
        self.load_trained_models()
        
        # Alert thresholds
        self.FAILURE_PROBABILITY_THRESHOLD = 0.7  # 70% probability
        self.RUL_CRITICAL_THRESHOLD = 3  # 3 days
        self.RUL_WARNING_THRESHOLD = 7   # 7 days
        
    def load_trained_models(self):
        """
        Load semua trained models dan components
        """
        print("🤖 Loading FlexoTwin Smart Maintenance Models...")
        print("=" * 60)
        
        try:
            # Load classification model
            self.classification_model = joblib.load('flexotwin_classification_random_forest.joblib')
            print("✅ Classification model loaded (Equipment Failure Prediction)")
            
            # Load regression model  
            self.regression_model = joblib.load('flexotwin_regression_random_forest.joblib')
            print("✅ Regression model loaded (RUL Estimation)")
            
            # Load scaler
            self.scaler = joblib.load('flexotwin_scaler.joblib')
            print("✅ Feature scaler loaded")
            
            # Load feature names
            self.feature_names = joblib.load('flexotwin_feature_names.joblib')
            print("✅ Feature names loaded")
            
            # Load model summary
            self.model_summary = joblib.load('flexotwin_model_summary.joblib')
            print("✅ Model performance summary loaded")
            
            # Display model performance
            clf_results = self.model_summary['classification']['results']['Random Forest']
            reg_results = self.model_summary['regression']['results']['Random Forest']
            
            print(f"\n📊 Model Performance Summary:")
            print(f"   🎯 Classification Model:")
            print(f"      - Accuracy: {clf_results['accuracy']:.4f}")
            print(f"      - AUC Score: {clf_results['auc_score']:.4f}")
            print(f"   📈 Regression Model:")
            print(f"      - R² Score: {reg_results['r2_score']:.4f}")
            print(f"      - RMSE: {reg_results['rmse']:.4f}")
            print(f"      - MAPE: {reg_results['mape']:.2f}%")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {str(e)}")
            print("💡 Please run model training first (03_model_development.py)")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
    
    def predict_single_record(self, input_features):
        """
        Predict untuk single production record
        
        Args:
            input_features (dict): Dictionary berisi feature values
            
        Returns:
            dict: Prediction results
        """
        if self.classification_model is None or self.regression_model is None:
            return {"error": "Models not loaded properly"}
        
        try:
            # Convert input ke DataFrame
            input_df = pd.DataFrame([input_features])
            
            # Ensure semua required features ada
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                # Fill missing features dengan median values atau 0
                for feature in missing_features:
                    input_df[feature] = 0
            
            # Reorder columns sesuai training
            input_df = input_df[self.feature_names]
            
            # Scale features (untuk models yang memerlukan scaling)
            input_scaled = self.scaler.transform(input_df)
            
            # Predictions
            failure_probability = self.classification_model.predict_proba(input_df)[0][1]
            failure_prediction = int(failure_probability > 0.5)
            estimated_rul = self.regression_model.predict(input_df)[0]
            
            # Generate alerts
            alerts = self.generate_alerts(failure_probability, estimated_rul)
            
            # Maintenance recommendations
            recommendations = self.generate_recommendations(failure_probability, estimated_rul)
            
            return {
                "failure_prediction": {
                    "will_fail": bool(failure_prediction),
                    "probability": float(failure_probability),
                    "confidence": "High" if failure_probability > 0.8 or failure_probability < 0.2 else "Medium"
                },
                "rul_estimation": {
                    "estimated_days": float(estimated_rul),
                    "category": self.categorize_rul(estimated_rul)
                },
                "alerts": alerts,
                "recommendations": recommendations,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, input_data_file):
        """
        Predict untuk batch data (CSV file)
        
        Args:
            input_data_file (str): Path ke CSV file
            
        Returns:
            DataFrame: Results dengan predictions
        """
        try:
            # Load data
            df = pd.read_csv(input_data_file)
            print(f"📊 Processing batch data: {len(df)} records")
            
            # Prepare features
            df_features = df.copy()
            
            # Ensure semua required features ada
            for feature in self.feature_names:
                if feature not in df_features.columns:
                    df_features[feature] = 0
            
            # Reorder columns
            df_features = df_features[self.feature_names]
            
            # Predictions
            failure_probabilities = self.classification_model.predict_proba(df_features)[:, 1]
            failure_predictions = (failure_probabilities > 0.5).astype(int)
            estimated_ruls = self.regression_model.predict(df_features)
            
            # Add predictions ke original dataframe
            df['Failure_Probability'] = failure_probabilities
            df['Failure_Prediction'] = failure_predictions
            df['Estimated_RUL'] = estimated_ruls
            df['RUL_Category'] = df['Estimated_RUL'].apply(self.categorize_rul)
            
            # Generate batch alerts
            high_risk_count = (failure_probabilities > self.FAILURE_PROBABILITY_THRESHOLD).sum()
            critical_rul_count = (estimated_ruls <= self.RUL_CRITICAL_THRESHOLD).sum()
            
            print(f"⚠️  Batch Analysis Results:")
            print(f"   - High failure risk records: {high_risk_count}")
            print(f"   - Critical RUL records: {critical_rul_count}")
            print(f"   - Average failure probability: {failure_probabilities.mean():.3f}")
            print(f"   - Average RUL: {estimated_ruls.mean():.1f} days")
            
            return df
            
        except Exception as e:
            print(f"❌ Batch prediction failed: {str(e)}")
            return None
    
    def generate_alerts(self, failure_probability, estimated_rul):
        """
        Generate maintenance alerts berdasarkan predictions
        """
        alerts = []
        
        # Failure probability alerts
        if failure_probability >= 0.9:
            alerts.append({
                "level": "CRITICAL",
                "type": "Equipment Failure Risk",
                "message": f"Very high failure probability ({failure_probability:.1%}). Immediate maintenance required!",
                "priority": 1
            })
        elif failure_probability >= self.FAILURE_PROBABILITY_THRESHOLD:
            alerts.append({
                "level": "WARNING", 
                "type": "Equipment Failure Risk",
                "message": f"High failure probability ({failure_probability:.1%}). Schedule maintenance soon.",
                "priority": 2
            })
        
        # RUL alerts
        if estimated_rul <= self.RUL_CRITICAL_THRESHOLD:
            alerts.append({
                "level": "CRITICAL",
                "type": "Remaining Useful Life",
                "message": f"Critical RUL: {estimated_rul:.1f} days. Immediate action needed!",
                "priority": 1
            })
        elif estimated_rul <= self.RUL_WARNING_THRESHOLD:
            alerts.append({
                "level": "WARNING",
                "type": "Remaining Useful Life", 
                "message": f"Low RUL: {estimated_rul:.1f} days. Plan maintenance within a week.",
                "priority": 2
            })
        
        # No alerts case
        if not alerts:
            alerts.append({
                "level": "NORMAL",
                "type": "System Status",
                "message": "Equipment operating normally. No immediate maintenance required.",
                "priority": 3
            })
        
        return alerts
    
    def generate_recommendations(self, failure_probability, estimated_rul):
        """
        Generate maintenance recommendations
        """
        recommendations = []
        
        if failure_probability >= 0.9 or estimated_rul <= 2:
            recommendations.extend([
                "Stop production immediately",
                "Perform comprehensive equipment inspection",
                "Check all critical components (hydraulics, mechanical parts)",
                "Replace worn parts before resuming production"
            ])
        elif failure_probability >= 0.7 or estimated_rul <= 5:
            recommendations.extend([
                "Schedule maintenance within 24-48 hours",
                "Inspect high-wear components",
                "Monitor performance closely",
                "Prepare replacement parts inventory"
            ])
        elif failure_probability >= 0.5 or estimated_rul <= 10:
            recommendations.extend([
                "Schedule preventive maintenance within a week",
                "Monitor OEE trends daily",
                "Check fluid levels and filters",
                "Review recent maintenance history"
            ])
        else:
            recommendations.extend([
                "Continue normal operations",
                "Maintain regular maintenance schedule",
                "Monitor performance metrics",
                "Keep spare parts inventory updated"
            ])
        
        return recommendations
    
    def categorize_rul(self, rul_days):
        """
        Categorize RUL into risk levels
        """
        if rul_days <= 3:
            return "Critical"
        elif rul_days <= 7:
            return "Warning"
        elif rul_days <= 15:
            return "Caution"
        else:
            return "Normal"
    
    def create_maintenance_dashboard(self, data_file='flexotwin_processed_data.csv'):
        """
        Create comprehensive maintenance dashboard
        """
        print("\n📊 Creating Maintenance Dashboard...")
        print("=" * 60)
        
        try:
            # Load recent data
            df = pd.read_csv(data_file)
            df['Posting_Date'] = pd.to_datetime(df['Posting_Date'])
            
            # Get predictions untuk recent data
            recent_data = df.tail(100)  # Last 100 records
            
            # Prepare features untuk prediction
            feature_data = recent_data.copy()
            for feature in self.feature_names:
                if feature not in feature_data.columns:
                    feature_data[feature] = 0
            
            feature_data = feature_data[self.feature_names]
            
            # Get predictions
            failure_probs = self.classification_model.predict_proba(feature_data)[:, 1]
            estimated_ruls = self.regression_model.predict(feature_data)
            
            # Create dashboard
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('FlexoTwin Smart Maintenance Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Failure Risk Trend
            axes[0, 0].plot(range(len(failure_probs)), failure_probs, color='red', linewidth=2)
            axes[0, 0].axhline(y=self.FAILURE_PROBABILITY_THRESHOLD, color='orange', linestyle='--', 
                              label=f'Warning Threshold ({self.FAILURE_PROBABILITY_THRESHOLD})')
            axes[0, 0].set_title('Equipment Failure Risk Trend')
            axes[0, 0].set_ylabel('Failure Probability')
            axes[0, 0].set_xlabel('Recent Production Records')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. RUL Trend
            axes[0, 1].plot(range(len(estimated_ruls)), estimated_ruls, color='blue', linewidth=2)
            axes[0, 1].axhline(y=self.RUL_CRITICAL_THRESHOLD, color='red', linestyle='--', 
                              label=f'Critical ({self.RUL_CRITICAL_THRESHOLD} days)')
            axes[0, 1].axhline(y=self.RUL_WARNING_THRESHOLD, color='orange', linestyle='--',
                              label=f'Warning ({self.RUL_WARNING_THRESHOLD} days)')
            axes[0, 1].set_title('Remaining Useful Life Trend')
            axes[0, 1].set_ylabel('RUL (days)')
            axes[0, 1].set_xlabel('Recent Production Records')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Risk Distribution
            risk_categories = pd.cut(failure_probs, bins=[0, 0.3, 0.7, 0.9, 1.0], 
                                   labels=['Low', 'Medium', 'High', 'Critical'])
            risk_counts = risk_categories.value_counts()
            
            colors = ['green', 'yellow', 'orange', 'red']
            axes[0, 2].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                          colors=colors, startangle=90)
            axes[0, 2].set_title('Current Risk Distribution')
            
            # 4. OEE Performance
            if 'OEE' in recent_data.columns:
                axes[1, 0].plot(range(len(recent_data)), recent_data['OEE'], color='purple', linewidth=2)
                axes[1, 0].axhline(y=0.6, color='red', linestyle='--', label='Poor Performance (60%)')
                axes[1, 0].axhline(y=0.85, color='green', linestyle='--', label='Target (85%)')
                axes[1, 0].set_title('OEE Performance Trend')
                axes[1, 0].set_ylabel('OEE Score')
                axes[1, 0].set_xlabel('Recent Production Records')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Downtime Analysis
            if 'Stop_Time' in recent_data.columns:
                axes[1, 1].hist(recent_data['Stop_Time'], bins=20, color='orange', alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Recent Downtime Distribution')
                axes[1, 1].set_xlabel('Downtime (minutes)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Alert Summary
            alert_summary = {
                'Critical': sum(1 for prob in failure_probs if prob >= 0.9) + sum(1 for rul in estimated_ruls if rul <= 3),
                'Warning': sum(1 for prob in failure_probs if 0.7 <= prob < 0.9) + sum(1 for rul in estimated_ruls if 3 < rul <= 7),
                'Normal': len(failure_probs) - sum(1 for prob in failure_probs if prob >= 0.7) - sum(1 for rul in estimated_ruls if rul <= 7)
            }
            
            axes[1, 2].bar(alert_summary.keys(), alert_summary.values(), 
                          color=['red', 'orange', 'green'], alpha=0.7)
            axes[1, 2].set_title('Current Alert Status')
            axes[1, 2].set_ylabel('Number of Records')
            
            # Add values on bars
            for i, (key, value) in enumerate(alert_summary.items()):
                axes[1, 2].text(i, value + 0.5, str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('flexotwin_maintenance_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Maintenance dashboard saved: 'flexotwin_maintenance_dashboard.png'")
            
            # Print summary statistics
            print(f"\n📊 Dashboard Summary:")
            print(f"   - Records analyzed: {len(recent_data)}")
            print(f"   - Average failure risk: {failure_probs.mean():.1%}")
            print(f"   - Average RUL: {estimated_ruls.mean():.1f} days")
            print(f"   - Critical alerts: {alert_summary['Critical']}")
            print(f"   - Warning alerts: {alert_summary['Warning']}")
            
        except Exception as e:
            print(f"❌ Dashboard creation failed: {str(e)}")
    
    def export_maintenance_report(self, data_file='flexotwin_processed_data.csv', output_file='maintenance_report.csv'):
        """
        Export comprehensive maintenance report
        """
        print(f"\n📋 Exporting Maintenance Report...")
        print("=" * 60)
        
        try:
            # Get batch predictions
            predictions_df = self.predict_batch(data_file)
            
            if predictions_df is not None:
                # Add additional analysis columns
                predictions_df['Risk_Level'] = predictions_df['Failure_Probability'].apply(
                    lambda x: 'Critical' if x >= 0.9 else 'High' if x >= 0.7 else 'Medium' if x >= 0.5 else 'Low'
                )
                
                predictions_df['Maintenance_Priority'] = predictions_df.apply(
                    lambda row: 1 if row['Failure_Probability'] >= 0.9 or row['Estimated_RUL'] <= 3
                               else 2 if row['Failure_Probability'] >= 0.7 or row['Estimated_RUL'] <= 7
                               else 3, axis=1
                )
                
                # Export to CSV
                predictions_df.to_csv(output_file, index=False)
                print(f"✅ Maintenance report exported: {output_file}")
                
                # Print summary
                high_risk = len(predictions_df[predictions_df['Risk_Level'].isin(['Critical', 'High'])])
                priority_1 = len(predictions_df[predictions_df['Maintenance_Priority'] == 1])
                
                print(f"📊 Report Summary:")
                print(f"   - Total records: {len(predictions_df):,}")
                print(f"   - High/Critical risk: {high_risk:,} ({high_risk/len(predictions_df)*100:.1f}%)")
                print(f"   - Priority 1 maintenance: {priority_1:,}")
                
                return predictions_df
            
        except Exception as e:
            print(f"❌ Report export failed: {str(e)}")
            return None

def demo_prediction_system():
    """
    Demo untuk FlexoTwin Smart Maintenance System
    """
    print("🚀 FlexoTwin Smart Maintenance System Demo")
    print("=" * 80)
    
    # Initialize system
    system = FlexoSmartMaintenanceSystem()
    
    if system.classification_model is None:
        print("❌ System not ready. Please train models first.")
        return
    
    print("\n🎯 Demo: Single Record Prediction")
    print("-" * 40)
    
    # Example input (bisa diganti dengan real-time data)
    sample_input = {
        'Machine': 1.0,
        'Shift': 1.0,
        'Confirm_Qty': 1500,
        'Scrab_Qty': 50,
        'Confirm_KG': 800.0,
        'Act_Confirm_KG': 780.0,
        'Scrab_KG': 20.0,
        'Stop_Time': 120,  # 2 hours downtime
        'OEE': 0.45,  # Poor OEE
        'Availability': 0.75,
        'Performance': 0.60,
        'Quality': 0.85,
        'Scrab_Rate': 0.15  # High scrap rate
    }
    
    # Get prediction
    result = system.predict_single_record(sample_input)
    
    if "error" not in result:
        print(f"📊 Prediction Results:")
        print(f"   🎯 Equipment Failure:")
        print(f"      - Will Fail: {'YES' if result['failure_prediction']['will_fail'] else 'NO'}")
        print(f"      - Probability: {result['failure_prediction']['probability']:.1%}")
        print(f"      - Confidence: {result['failure_prediction']['confidence']}")
        
        print(f"   📈 Remaining Useful Life:")
        print(f"      - Estimated: {result['rul_estimation']['estimated_days']:.1f} days")
        print(f"      - Category: {result['rul_estimation']['category']}")
        
        print(f"   ⚠️  Alerts:")
        for alert in result['alerts']:
            print(f"      - [{alert['level']}] {alert['message']}")
        
        print(f"   💡 Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"      {i}. {rec}")
    
    # Create dashboard
    print(f"\n📊 Creating Maintenance Dashboard...")
    system.create_maintenance_dashboard()
    
    # Export report
    print(f"\n📋 Exporting Maintenance Report...")
    system.export_maintenance_report()
    
    print(f"\n🎉 Demo completed! Check generated files:")
    print(f"   - flexotwin_maintenance_dashboard.png")
    print(f"   - maintenance_report.csv")

if __name__ == "__main__":
    demo_prediction_system()