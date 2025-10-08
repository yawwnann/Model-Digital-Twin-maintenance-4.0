"""
FlexoTwin Smart Maintenance 4.0
Model Development & Training Script

Tujuan: 
1. Klasifikasi - Prediksi Equipment Failure
2. Regresi - Estimasi Remaining Useful Life (RUL)
3. Model evaluation dan optimization
4. Feature importance analysis

Dibuat untuk: Proyek Skripsi Teknik Industri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            mean_squared_error, r2_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class FlexoModelDeveloper:
    def __init__(self, data_file='flexotwin_processed_data.csv'):
        """
        Inisialisasi Model Developer untuk FlexoTwin
        
        Args:
            data_file (str): Path ke processed data file
        """
        self.data_file = data_file
        self.df = None
        self.X = None
        self.y_classification = None
        self.y_regression = None
        
        # Model containers
        self.classification_models = {}
        self.regression_models = {}
        self.best_classification_model = None
        self.best_regression_model = None
        
        # Scalers
        self.scaler = StandardScaler()
        
        # Results
        self.classification_results = {}
        self.regression_results = {}
        
    def load_processed_data(self):
        """
        Load processed data dari CSV
        """
        print("📂 Loading Processed Data...")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded successfully")
            print(f"📊 Shape: {self.df.shape}")
            
            # Convert Posting_Date to datetime if it's not already
            if 'Posting_Date' in self.df.columns:
                self.df['Posting_Date'] = pd.to_datetime(self.df['Posting_Date'], errors='coerce')
                print(f"📅 Date range: {self.df['Posting_Date'].min()} to {self.df['Posting_Date'].max()}")
            
            # Check for missing values
            missing_summary = self.df.isnull().sum()
            missing_cols = missing_summary[missing_summary > 0]
            
            if len(missing_cols) > 0:
                print(f"⚠️ Missing values found in {len(missing_cols)} columns")
                for col, count in missing_cols.items():
                    print(f"   - {col}: {count} missing ({count/len(self.df)*100:.1f}%)")
            else:
                print("✅ No missing values found")
                
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def prepare_features_and_targets(self):
        """
        Prepare features dan target variables untuk modeling
        """
        print("\n🎯 Preparing Features and Targets...")
        print("=" * 60)
        
        if self.df is None:
            print("❌ No data loaded")
            return None, None, None
        
        # Define features untuk training (exclude target dan identifier columns)
        exclude_columns = [
            'Posting_Date', 'Month', 'Year', 'Work_Center', 'Prod_Order',
            'Start_Date', 'Start_Time', 'Finish_Time', 'Finish_Date',
            'Scrab_Description', 'Break_Description',
            # Target variables
            'Equipment_Failure', 'High_Downtime', 'High_Scrap_Rate', 
            'Poor_OEE', 'Failure_Severity', 'Estimated_RUL'
        ]
        
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        # Prepare feature matrix X
        self.X = self.df[feature_columns].copy()
        
        # Handle categorical variables
        categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            print(f"🔄 Encoding {len(categorical_columns)} categorical columns...")
            
            le = LabelEncoder()
            for col in categorical_columns:
                self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # Handle any remaining missing values
        if self.X.isnull().sum().sum() > 0:
            print("🔄 Imputing missing values...")
            imputer = SimpleImputer(strategy='median')
            self.X = pd.DataFrame(imputer.fit_transform(self.X), 
                                columns=self.X.columns, index=self.X.index)
        
        # Define target variables
        self.y_classification = self.df['Equipment_Failure']  # Binary classification
        self.y_regression = self.df['Estimated_RUL']  # Continuous regression
        
        print(f"✅ Feature preparation completed")
        print(f"📊 Features: {self.X.shape[1]} columns")
        print(f"🎯 Classification target - Equipment Failure:")
        print(f"   - Class 0 (Normal): {(self.y_classification == 0).sum()} ({(self.y_classification == 0).mean()*100:.1f}%)")
        print(f"   - Class 1 (Failure): {(self.y_classification == 1).sum()} ({(self.y_classification == 1).mean()*100:.1f}%)")
        print(f"🎯 Regression target - Estimated RUL:")
        print(f"   - Mean: {self.y_regression.mean():.2f} days")
        print(f"   - Range: {self.y_regression.min():.2f} - {self.y_regression.max():.2f} days")
        
        return self.X, self.y_classification, self.y_regression
    
    def train_classification_models(self):
        """
        Train multiple classification models untuk failure prediction
        """
        print("\n🏗️ Training Classification Models...")
        print("=" * 60)
        
        if self.X is None or self.y_classification is None:
            print("❌ Features and targets not prepared")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_classification, test_size=0.2, random_state=42, stratify=self.y_classification
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        print(f"🔄 Training {len(models)} classification models...")
        
        for name, model in models.items():
            print(f"\n📈 Training {name}...")
            
            try:
                # Use scaled data untuk SVM dan Logistic Regression
                if name in ['SVM', 'Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = model.score(X_test_scaled if name in ['SVM', 'Logistic Regression'] else X_test, y_test)
                auc_score = roc_auc_score(y_test, y_prob)
                
                # Store model dan results
                self.classification_models[name] = model
                self.classification_results[name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'y_test': y_test
                }
                
                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   ✅ AUC Score: {auc_score:.4f}")
                
                # Cross-validation
                if name in ['SVM', 'Logistic Regression']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
        
        # Identify best model
        best_auc = 0
        for name, results in self.classification_results.items():
            if results['auc_score'] > best_auc:
                best_auc = results['auc_score']
                self.best_classification_model = name
        
        print(f"\n🏆 Best Classification Model: {self.best_classification_model} (AUC: {best_auc:.4f})")
        
        return self.classification_models
    
    def train_regression_models(self):
        """
        Train multiple regression models untuk RUL prediction
        """
        print("\n🏗️ Training Regression Models...")
        print("=" * 60)
        
        if self.X is None or self.y_regression is None:
            print("❌ Features and targets not prepared")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
        
        print(f"🔄 Training {len(models)} regression models...")
        
        for name, model in models.items():
            print(f"\n📈 Training {name}...")
            
            try:
                # Use scaled data untuk Linear Regression dan SVR
                if name in ['Linear Regression', 'SVR']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
                
                # Store model dan results
                self.regression_models[name] = model
                self.regression_results[name] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                print(f"   ✅ R² Score: {r2:.4f}")
                print(f"   ✅ RMSE: {rmse:.4f}")
                print(f"   ✅ MAE: {mae:.4f}")
                print(f"   ✅ MAPE: {mape:.2f}%")
                
                # Cross-validation
                if name in ['Linear Regression', 'SVR']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                print(f"   ✅ CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
        
        # Identify best model
        best_r2 = -999
        for name, results in self.regression_results.items():
            if results['r2_score'] > best_r2:
                best_r2 = results['r2_score']
                self.best_regression_model = name
        
        print(f"\n🏆 Best Regression Model: {self.best_regression_model} (R²: {best_r2:.4f})")
        
        return self.regression_models
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance dari tree-based models
        """
        print("\n📊 Feature Importance Analysis...")
        print("=" * 60)
        
        # Classification feature importance
        if self.best_classification_model in ['Random Forest', 'Gradient Boosting']:
            clf_model = self.classification_models[self.best_classification_model]
            
            if hasattr(clf_model, 'feature_importances_'):
                clf_importance = pd.DataFrame({
                    'feature': self.X.columns,
                    'importance': clf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"🎯 Top 10 Features for Classification ({self.best_classification_model}):")
                for i, (_, row) in enumerate(clf_importance.head(10).iterrows(), 1):
                    print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Regression feature importance  
        if self.best_regression_model in ['Random Forest', 'Gradient Boosting']:
            reg_model = self.regression_models[self.best_regression_model]
            
            if hasattr(reg_model, 'feature_importances_'):
                reg_importance = pd.DataFrame({
                    'feature': self.X.columns,
                    'importance': reg_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n📈 Top 10 Features for Regression ({self.best_regression_model}):")
                for i, (_, row) in enumerate(reg_importance.head(10).iterrows(), 1):
                    print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Create feature importance visualization
        self.create_feature_importance_plot()
    
    def create_feature_importance_plot(self):
        """
        Create feature importance visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Classification feature importance
        if (self.best_classification_model in ['Random Forest', 'Gradient Boosting'] and 
            hasattr(self.classification_models[self.best_classification_model], 'feature_importances_')):
            
            clf_model = self.classification_models[self.best_classification_model]
            clf_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': clf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[0].barh(range(len(clf_importance)), clf_importance['importance'], color='steelblue')
            axes[0].set_yticks(range(len(clf_importance)))
            axes[0].set_yticklabels(clf_importance['feature'])
            axes[0].set_xlabel('Feature Importance')
            axes[0].set_title(f'Top 15 Features - Classification ({self.best_classification_model})')
            axes[0].invert_yaxis()
        
        # Regression feature importance
        if (self.best_regression_model in ['Random Forest', 'Gradient Boosting'] and 
            hasattr(self.regression_models[self.best_regression_model], 'feature_importances_')):
            
            reg_model = self.regression_models[self.best_regression_model]
            reg_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': reg_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[1].barh(range(len(reg_importance)), reg_importance['importance'], color='orange')
            axes[1].set_yticks(range(len(reg_importance)))
            axes[1].set_yticklabels(reg_importance['feature'])
            axes[1].set_xlabel('Feature Importance')
            axes[1].set_title(f'Top 15 Features - Regression ({self.best_regression_model})')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('flexotwin_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance plot saved: 'flexotwin_feature_importance.png'")
    
    def create_model_evaluation_plots(self):
        """
        Create comprehensive model evaluation plots
        """
        print("\n📊 Creating Model Evaluation Plots...")
        print("=" * 60)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Classification Results Comparison
        plt.subplot(3, 3, 1)
        models = list(self.classification_results.keys())
        accuracies = [self.classification_results[model]['accuracy'] for model in models]
        auc_scores = [self.classification_results[model]['auc_score'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
        plt.bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8, color='orange')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Classification Models Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        
        # 2. Regression Results Comparison
        plt.subplot(3, 3, 2)
        models = list(self.regression_results.keys())
        r2_scores = [self.regression_results[model]['r2_score'] for model in models]
        rmse_scores = [self.regression_results[model]['rmse'] for model in models]
        
        # Normalize RMSE for comparison
        max_rmse = max(rmse_scores)
        rmse_normalized = [1 - (rmse/max_rmse) for rmse in rmse_scores]
        
        x = np.arange(len(models))
        plt.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8, color='green')
        plt.bar(x + width/2, rmse_normalized, width, label='1 - Normalized RMSE', alpha=0.8, color='red')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Regression Models Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        
        # 3. Confusion Matrix untuk best classification model
        if self.best_classification_model:
            plt.subplot(3, 3, 3)
            y_test = self.classification_results[self.best_classification_model]['y_test']
            y_pred = self.classification_results[self.best_classification_model]['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
            plt.title(f'Confusion Matrix - {self.best_classification_model}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # 4. Actual vs Predicted untuk best regression model
        if self.best_regression_model:
            plt.subplot(3, 3, 4)
            y_test = self.regression_results[self.best_regression_model]['y_test']
            y_pred = self.regression_results[self.best_regression_model]['y_pred']
            
            plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual RUL')
            plt.ylabel('Predicted RUL')
            plt.title(f'Actual vs Predicted RUL - {self.best_regression_model}')
            
            # Add R² score
            r2 = self.regression_results[self.best_regression_model]['r2_score']
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5-9. Individual model performance plots can be added here
        
        plt.tight_layout()
        plt.savefig('flexotwin_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Model evaluation plots saved: 'flexotwin_model_evaluation.png'")
    
    def save_models(self):
        """
        Save trained models untuk deployment
        """
        print("\n💾 Saving Trained Models...")
        print("=" * 60)
        
        # Save best models
        if self.best_classification_model:
            model_filename = f'flexotwin_classification_{self.best_classification_model.lower().replace(" ", "_")}.joblib'
            joblib.dump(self.classification_models[self.best_classification_model], model_filename)
            print(f"✅ Classification model saved: {model_filename}")
        
        if self.best_regression_model:
            model_filename = f'flexotwin_regression_{self.best_regression_model.lower().replace(" ", "_")}.joblib'
            joblib.dump(self.regression_models[self.best_regression_model], model_filename)
            print(f"✅ Regression model saved: {model_filename}")
        
        # Save scaler
        joblib.dump(self.scaler, 'flexotwin_scaler.joblib')
        print("✅ Feature scaler saved: flexotwin_scaler.joblib")
        
        # Save feature names
        feature_names = list(self.X.columns)
        joblib.dump(feature_names, 'flexotwin_feature_names.joblib')
        print("✅ Feature names saved: flexotwin_feature_names.joblib")
        
        # Save model performance summary
        summary = {
            'classification': {
                'best_model': self.best_classification_model,
                'results': self.classification_results
            },
            'regression': {
                'best_model': self.best_regression_model,
                'results': self.regression_results
            }
        }
        
        joblib.dump(summary, 'flexotwin_model_summary.joblib')
        print("✅ Model summary saved: flexotwin_model_summary.joblib")

def main():
    """
    Main function untuk model development
    """
    print("🚀 FlexoTwin Model Development & Training")
    print("=" * 80)
    
    # Initialize model developer
    developer = FlexoModelDeveloper()
    
    # 1. Load processed data
    data = developer.load_processed_data()
    
    if data is not None:
        # 2. Prepare features and targets
        X, y_clf, y_reg = developer.prepare_features_and_targets()
        
        if X is not None:
            # 3. Train classification models
            developer.train_classification_models()
            
            # 4. Train regression models
            developer.train_regression_models()
            
            # 5. Analyze feature importance
            developer.analyze_feature_importance()
            
            # 6. Create evaluation plots
            developer.create_model_evaluation_plots()
            
            # 7. Save models
            developer.save_models()
            
            print("\n🎉 Model Development Completed Successfully!")
            print("📋 Summary:")
            print(f"   🏆 Best Classification Model: {developer.best_classification_model}")
            print(f"   🏆 Best Regression Model: {developer.best_regression_model}")
            print("\n📋 Next Steps:")
            print("   1. Review model performance and feature importance")
            print("   2. Consider hyperparameter tuning for best models") 
            print("   3. Develop prediction interface/dashboard")
            print("   4. Implement model deployment strategy")
            
        else:
            print("❌ Feature preparation failed")
    else:
        print("❌ Data loading failed")

if __name__ == "__main__":
    main()