"""
FlexoTwin Smart Maintenance - Model Prediksi Maintenance
Tujuan Spesifik:
1. Prediksi kapan mesin/komponen akan rusak lagi
2. Rekomendasi kapan sebaiknya melakukan maintenance
3. Sistem alert untuk mencegah kerusakan

Untuk integrasi dengan website
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class MaintenancePredictionSystem:
    def __init__(self):
        """
        Sistem Prediksi Maintenance untuk Mesin Flexo 104
        """
        self.failure_model = None      # Model prediksi kapan rusak
        self.maintenance_model = None  # Model rekomendasi maintenance
        self.feature_names = None
        self.is_trained = False
        
    def load_and_prepare_data(self):
        """
        Load data dan persiapkan untuk training model maintenance
        """
        print("📂 Loading Data untuk Maintenance Prediction...")
        print("=" * 60)
        
        try:
            # Load processed data
            df = pd.read_csv('flexotwin_processed_data.csv')
            print(f"✅ Data loaded: {len(df)} records")
            
            # Create maintenance-specific features dan targets
            df = self._create_maintenance_features(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def _create_maintenance_features(self, df):
        """
        Create features khusus untuk maintenance prediction
        """
        print("🔧 Creating Maintenance-Specific Features...")
        
        # 1. FAILURE TIME PREDICTION FEATURES
        # Hitung days until next failure berdasarkan degradation pattern
        df['Days_Since_Last_Failure'] = 0
        df['Performance_Decline_Rate'] = df['Performance'].rolling(window=7).apply(
            lambda x: (x.iloc[0] - x.iloc[-1]) / 7 if len(x) >= 2 else 0
        ).fillna(0)
        
        # 2. MAINTENANCE URGENCY SCORE (0-100)
        # Kombinasi multiple factors untuk urgency
        df['Maintenance_Urgency'] = (
            (1 - df['OEE']) * 30 +                    # OEE impact (30%)
            (df['Downtime'] / df['Downtime'].max()) * 25 +  # Downtime impact (25%)
            (df['Scrab_Rate']) * 20 +                 # Quality impact (20%)
            (df['Performance_Decline_Rate']) * 25     # Degradation trend (25%)
        ) * 100
        
        # Cap at 100
        df['Maintenance_Urgency'] = np.clip(df['Maintenance_Urgency'], 0, 100)
        
        # 3. DAYS UNTIL MAINTENANCE NEEDED
        # Based on degradation rate and current performance
        df['Days_Until_Maintenance'] = np.where(
            df['Performance_Decline_Rate'] > 0,
            np.clip(
                (df['Performance'] - 0.3) / (df['Performance_Decline_Rate'] + 0.001),  # Before performance drops to 30%
                1, 30
            ),
            30  # Max 30 days if no decline detected
        )
        
        # 4. FAILURE RISK CATEGORIES
        df['Failure_Risk_Level'] = pd.cut(
            df['Maintenance_Urgency'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        # 5. MAINTENANCE TYPE RECOMMENDATION
        df['Maintenance_Type'] = 'Routine'
        df.loc[df['Maintenance_Urgency'] > 75, 'Maintenance_Type'] = 'Emergency'
        df.loc[(df['Maintenance_Urgency'] > 50) & (df['Maintenance_Urgency'] <= 75), 'Maintenance_Type'] = 'Preventive'
        df.loc[(df['Maintenance_Urgency'] > 25) & (df['Maintenance_Urgency'] <= 50), 'Maintenance_Type'] = 'Scheduled'
        
        # 6. COMPONENT-SPECIFIC RISK
        # Berdasarkan failure patterns yang terlihat
        df['Hydraulic_Risk'] = np.where(df['Stop_Time'] > 200, 1, 0)  # Long downtimes suggest hydraulic issues
        df['Print_Quality_Risk'] = np.where(df['Scrab_Rate'] > 0.15, 1, 0)  # High scrap suggests print issues
        df['Mechanical_Risk'] = np.where(df['Performance'] < 0.5, 1, 0)  # Low performance suggests mechanical issues
        
        print(f"✅ Maintenance features created")
        print(f"   - Maintenance Urgency range: {df['Maintenance_Urgency'].min():.1f} - {df['Maintenance_Urgency'].max():.1f}")
        print(f"   - Days Until Maintenance range: {df['Days_Until_Maintenance'].min():.1f} - {df['Days_Until_Maintenance'].max():.1f}")
        print(f"   - Risk Level distribution:")
        print(df['Failure_Risk_Level'].value_counts().to_string())
        
        return df
    
    def train_maintenance_models(self, df):
        """
        Train models untuk maintenance prediction
        """
        print("\n🤖 Training Maintenance Prediction Models...")
        print("=" * 60)
        
        # Select features untuk training
        feature_cols = [
            'OEE', 'Availability', 'Performance', 'Quality',
            'Downtime', 'Scrab_Rate', 'Total_Production',
            'Performance_Decline_Rate', 'Days_Since_Last_Failure',
            'Hydraulic_Risk', 'Print_Quality_Risk', 'Mechanical_Risk',
            'Shift', 'Day_of_Week', 'Month_Num'
        ]
        
        # Ensure columns exist
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Target 1: Days until maintenance needed (Regression)
        y_days = df['Days_Until_Maintenance'].fillna(30)
        
        # Target 2: Maintenance urgency classification
        y_urgency = df['Maintenance_Urgency'].fillna(0)
        urgency_categories = pd.cut(y_urgency, bins=[0, 25, 50, 75, 100], labels=[0, 1, 2, 3])
        
        # Split data
        X_train, X_test, y_days_train, y_days_test = train_test_split(
            X, y_days, test_size=0.2, random_state=42
        )
        
        _, _, y_urgency_train, y_urgency_test = train_test_split(
            X, urgency_categories, test_size=0.2, random_state=42
        )
        
        # Train Model 1: Days Until Maintenance (Regression)
        print("📈 Training Days-Until-Maintenance Predictor...")
        self.maintenance_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.maintenance_model.fit(X_train, y_days_train)
        
        days_pred = self.maintenance_model.predict(X_test)
        days_r2 = r2_score(y_days_test, days_pred)
        days_mae = np.mean(np.abs(y_days_test - days_pred))
        
        print(f"   ✅ R² Score: {days_r2:.4f}")
        print(f"   ✅ Mean Absolute Error: {days_mae:.2f} days")
        
        # Train Model 2: Failure Risk Classification
        print("📊 Training Failure Risk Classifier...")
        self.failure_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        # Remove NaN values untuk classification
        mask = ~pd.isna(y_urgency_train)
        X_train_clean = X_train[mask]
        y_urgency_train_clean = y_urgency_train[mask]
        
        if len(y_urgency_train_clean) > 0:
            self.failure_model.fit(X_train_clean, y_urgency_train_clean)
            
            # Test predictions
            mask_test = ~pd.isna(y_urgency_test)
            if mask_test.sum() > 0:
                urgency_pred = self.failure_model.predict(X_test[mask_test])
                urgency_accuracy = accuracy_score(y_urgency_test[mask_test], urgency_pred)
                print(f"   ✅ Classification Accuracy: {urgency_accuracy:.4f}")
        
        # Store feature names
        self.feature_names = available_features
        self.is_trained = True
        
        print("🎉 Maintenance models training completed!")
        
        return self.maintenance_model, self.failure_model
    
    def predict_maintenance_schedule(self, input_data):
        """
        Prediksi jadwal maintenance untuk input data
        
        Args:
            input_data (dict): Current machine condition data
            
        Returns:
            dict: Maintenance predictions dan recommendations
        """
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        try:
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features ada
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[self.feature_names]
            
            # Predictions
            days_until_maintenance = self.maintenance_model.predict(input_df)[0]
            risk_level = self.failure_model.predict(input_df)[0] if self.failure_model else 0
            
            # Calculate maintenance urgency score
            urgency_score = self._calculate_urgency_score(input_data)
            
            # Generate recommendations
            recommendations = self._generate_maintenance_recommendations(
                days_until_maintenance, risk_level, urgency_score
            )
            
            # Maintenance schedule
            today = datetime.now()
            maintenance_date = today + timedelta(days=int(days_until_maintenance))
            
            return {
                "prediction_date": today.strftime("%Y-%m-%d %H:%M:%S"),
                "days_until_maintenance": round(days_until_maintenance, 1),
                "recommended_maintenance_date": maintenance_date.strftime("%Y-%m-%d"),
                "risk_level": self._get_risk_level_name(risk_level),
                "urgency_score": round(urgency_score, 1),
                "maintenance_type": recommendations["type"],
                "priority": recommendations["priority"],
                "recommended_actions": recommendations["actions"],
                "components_to_check": recommendations["components"],
                "estimated_downtime": recommendations["downtime"],
                "cost_impact": recommendations["cost"]
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _calculate_urgency_score(self, data):
        """Calculate urgency score from current data"""
        oee = data.get('OEE', 0.5)
        downtime = data.get('Downtime', 0)
        scrap_rate = data.get('Scrab_Rate', 0)
        performance_decline = data.get('Performance_Decline_Rate', 0)
        
        urgency = (
            (1 - oee) * 30 +
            min(downtime / 480, 1) * 25 +  # Normalize downtime to 8 hours max
            scrap_rate * 20 +
            performance_decline * 25
        )
        
        return min(urgency * 100, 100)
    
    def _get_risk_level_name(self, risk_code):
        """Convert risk code to name"""
        risk_names = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        return risk_names.get(risk_code, "Unknown")
    
    def _generate_maintenance_recommendations(self, days_until, risk_level, urgency_score):
        """Generate detailed maintenance recommendations"""
        
        if urgency_score >= 80 or days_until <= 2:
            return {
                "type": "Emergency Maintenance",
                "priority": "CRITICAL - Immediate Action Required",
                "actions": [
                    "Stop production immediately",
                    "Perform comprehensive equipment inspection",
                    "Check hydraulic system pressure and fluid levels",
                    "Inspect all mechanical components for wear",
                    "Replace critical parts before resuming production"
                ],
                "components": ["Hydraulic System", "Print Cylinders", "Drive Motors", "Sensors"],
                "downtime": "4-8 hours",
                "cost": "High (Rp 50-100 juta)"
            }
        
        elif urgency_score >= 60 or days_until <= 5:
            return {
                "type": "Preventive Maintenance",
                "priority": "HIGH - Schedule within 48 hours",
                "actions": [
                    "Schedule maintenance within 2 days",
                    "Prepare replacement parts inventory",
                    "Inspect high-wear components",
                    "Check fluid levels and filters",
                    "Monitor performance closely until maintenance"
                ],
                "components": ["Filters", "Belts", "Bearings", "Fluid Systems"],
                "downtime": "2-4 hours",
                "cost": "Medium (Rp 20-50 juta)"
            }
        
        elif urgency_score >= 40 or days_until <= 10:
            return {
                "type": "Scheduled Maintenance",
                "priority": "MEDIUM - Plan within a week",
                "actions": [
                    "Schedule maintenance within next week",
                    "Order standard replacement parts",
                    "Review recent maintenance history",
                    "Plan downtime during low production periods",
                    "Prepare maintenance team and tools"
                ],
                "components": ["Routine Inspection", "Lubrication", "Calibration"],
                "downtime": "1-2 hours",
                "cost": "Low (Rp 5-20 juta)"
            }
        
        else:
            return {
                "type": "Routine Maintenance",
                "priority": "LOW - Continue regular schedule",
                "actions": [
                    "Continue normal operations",
                    "Follow regular maintenance schedule",
                    "Monitor performance metrics daily",
                    "Keep spare parts inventory updated",
                    "Document any unusual observations"
                ],
                "components": ["General Inspection", "Basic Cleaning", "Documentation"],
                "downtime": "30-60 minutes",
                "cost": "Very Low (< Rp 5 juta)"
            }
    
    def create_maintenance_dashboard(self):
        """
        Create dashboard untuk maintenance planning
        """
        try:
            df = pd.read_csv('flexotwin_processed_data.csv')
            
            # Load atau create maintenance features
            df = self._create_maintenance_features(df)
            
            # Create dashboard
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Maintenance Prediction Dashboard - FlexoTwin', fontsize=16, fontweight='bold')
            
            # 1. Days Until Maintenance Distribution
            axes[0, 0].hist(df['Days_Until_Maintenance'], bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(df['Days_Until_Maintenance'].mean(), color='red', linestyle='--', 
                              label=f'Average: {df["Days_Until_Maintenance"].mean():.1f} days')
            axes[0, 0].set_title('Days Until Maintenance Needed')
            axes[0, 0].set_xlabel('Days')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Maintenance Urgency Score
            axes[0, 1].hist(df['Maintenance_Urgency'], bins=25, color='red', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(df['Maintenance_Urgency'].mean(), color='blue', linestyle='--',
                              label=f'Average: {df["Maintenance_Urgency"].mean():.1f}')
            axes[0, 1].set_title('Maintenance Urgency Distribution')
            axes[0, 1].set_xlabel('Urgency Score (0-100)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Risk Level Distribution
            risk_counts = df['Failure_Risk_Level'].value_counts()
            colors = ['green', 'yellow', 'orange', 'red']
            axes[0, 2].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                          colors=colors[:len(risk_counts)], startangle=90)
            axes[0, 2].set_title('Risk Level Distribution')
            
            # 4. Maintenance Type Recommendations
            maint_counts = df['Maintenance_Type'].value_counts()
            axes[1, 0].bar(maint_counts.index, maint_counts.values, 
                          color=['green', 'yellow', 'orange', 'red'][:len(maint_counts)])
            axes[1, 0].set_title('Recommended Maintenance Types')
            axes[1, 0].set_ylabel('Number of Records')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. Component Risk Analysis
            component_risks = df[['Hydraulic_Risk', 'Print_Quality_Risk', 'Mechanical_Risk']].sum()
            axes[1, 1].bar(component_risks.index, component_risks.values, color='purple', alpha=0.7)
            axes[1, 1].set_title('Component Risk Analysis')
            axes[1, 1].set_ylabel('High Risk Events')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 6. Maintenance Urgency vs OEE
            scatter = axes[1, 2].scatter(df['OEE'], df['Maintenance_Urgency'], 
                                       c=df['Days_Until_Maintenance'], cmap='RdYlGn_r', alpha=0.6)
            axes[1, 2].set_xlabel('OEE Score')
            axes[1, 2].set_ylabel('Maintenance Urgency')
            axes[1, 2].set_title('Urgency vs Performance Relationship')
            plt.colorbar(scatter, ax=axes[1, 2], label='Days Until Maintenance')
            
            plt.tight_layout()
            plt.savefig('maintenance_prediction_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Maintenance dashboard saved: 'maintenance_prediction_dashboard.png'")
            
        except Exception as e:
            print(f"❌ Dashboard creation failed: {str(e)}")
    
    def save_models(self):
        """Save maintenance prediction models"""
        if self.is_trained:
            joblib.dump(self.maintenance_model, 'maintenance_days_predictor.joblib')
            joblib.dump(self.failure_model, 'maintenance_risk_classifier.joblib')
            joblib.dump(self.feature_names, 'maintenance_feature_names.joblib')
            
            print("✅ Maintenance models saved:")
            print("   - maintenance_days_predictor.joblib")
            print("   - maintenance_risk_classifier.joblib")
            print("   - maintenance_feature_names.joblib")
        else:
            print("❌ No trained models to save")

def main():
    """
    Main function untuk training dan testing maintenance prediction system
    """
    print("🔧 FlexoTwin Maintenance Prediction System")
    print("=" * 70)
    print("Tujuan:")
    print("1. 📅 Prediksi kapan mesin/komponen akan rusak")
    print("2. 🔧 Rekomendasi jadwal maintenance yang optimal")
    print("3. ⚠️  Alert system untuk pencegahan kerusakan")
    print("=" * 70)
    
    # Initialize system
    system = MaintenancePredictionSystem()
    
    # 1. Load dan prepare data
    df = system.load_and_prepare_data()
    
    if df is not None:
        # 2. Train models
        system.train_maintenance_models(df)
        
        # 3. Demo prediction
        print("\n🎯 Demo: Maintenance Prediction")
        print("=" * 50)
        
        # Example current machine condition
        current_condition = {
            'OEE': 0.15,          # Low OEE - concerning
            'Availability': 0.8,   # Good availability
            'Performance': 0.2,    # Very poor performance
            'Quality': 0.95,       # Good quality
            'Downtime': 180,       # 3 hours downtime
            'Scrab_Rate': 0.05,    # Low scrap rate
            'Total_Production': 1000,
            'Performance_Decline_Rate': 0.02,  # Declining 2% per day
            'Days_Since_Last_Failure': 15,
            'Shift': 1,
            'Day_of_Week': 1,      # Monday
            'Month_Num': 10        # October
        }
        
        # Get prediction
        prediction = system.predict_maintenance_schedule(current_condition)
        
        if "error" not in prediction:
            print(f"📊 MAINTENANCE PREDICTION RESULTS:")
            print(f"=" * 40)
            print(f"📅 Prediction Date: {prediction['prediction_date']}")
            print(f"⏰ Days Until Maintenance: {prediction['days_until_maintenance']} days")
            print(f"📆 Recommended Date: {prediction['recommended_maintenance_date']}")
            print(f"🚨 Risk Level: {prediction['risk_level']}")
            print(f"📈 Urgency Score: {prediction['urgency_score']}/100")
            print(f"🔧 Maintenance Type: {prediction['maintenance_type']}")
            print(f"⚡ Priority: {prediction['priority']}")
            print(f"⏱️  Estimated Downtime: {prediction['estimated_downtime']}")
            print(f"💰 Cost Impact: {prediction['cost_impact']}")
            
            print(f"\n📋 RECOMMENDED ACTIONS:")
            for i, action in enumerate(prediction['recommended_actions'], 1):
                print(f"   {i}. {action}")
            
            print(f"\n🔧 COMPONENTS TO CHECK:")
            for component in prediction['components_to_check']:
                print(f"   • {component}")
        
        # 4. Create dashboard
        system.create_maintenance_dashboard()
        
        # 5. Save models
        system.save_models()
        
        print(f"\n🎉 Maintenance Prediction System Ready!")
        print(f"💡 Next: Integrate dengan website untuk real-time monitoring")
        
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    main()