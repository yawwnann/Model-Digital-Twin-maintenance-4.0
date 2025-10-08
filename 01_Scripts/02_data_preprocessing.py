"""
FlexoTwin Smart Maintenance 4.0
Data Preprocessing & Feature Engineering Script

Tujuan: 
1. Membersihkan dan standardisasi data
2. Menghitung metrik OEE (Overall Equipment Effectiveness)
3. Feature engineering untuk predictive maintenance
4. Integrasi data produksi dan perawatan

Dibuat untuk: Proyek Skripsi Teknik Industri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
 
class FlexoDataProcessor:
    def __init__(self, data_path):
        """
        Inisialisasi Data Processor untuk FlexoTwin
        
        Args:
            data_path (str): Path ke folder data
        """
        self.data_path = data_path
        self.production_data = {}
        self.maintenance_data = None
        self.processed_production = None
        self.integrated_data = None
        
    def load_and_clean_production_data(self):
        """
        Load dan bersihkan data produksi
        """
        print("🔄 Loading & Cleaning Production Data...")
        print("=" * 60)
        
        from pathlib import Path
        data_path = Path(self.data_path)
        production_files = list(data_path.glob("Produksi Bulan *.xlsx"))
        
        all_production_data = []
        
        for file in production_files:
            try:
                month_name = file.stem.replace("Produksi Bulan ", "").replace(" 2025", "")
                print(f"📁 Processing: {month_name}")
                
                # Load data
                df = pd.read_excel(file)
                
                # Filter C_FL104
                if 'Work Center' in df.columns:
                    df = df[df['Work Center'] == 'C_FL104']
                
                # Add month identifier
                df['Month'] = month_name
                df['Year'] = 2025
                
                # Clean dan standardisasi kolom
                df = self._clean_production_columns(df, month_name)
                
                all_production_data.append(df)
                print(f"✅ {month_name}: {len(df)} records processed")
                
            except Exception as e:
                print(f"❌ Error processing {file.name}: {str(e)}")
        
        # Combine semua data
        if all_production_data:
            self.processed_production = pd.concat(all_production_data, ignore_index=True)
            print(f"\n🎉 Combined production data: {len(self.processed_production)} total records")
            
            # Sort by date
            self.processed_production = self.processed_production.sort_values('Posting_Date').reset_index(drop=True)
        
        return self.processed_production
    
    def _clean_production_columns(self, df, month_name):
        """
        Bersihkan dan standardisasi kolom produksi
        """
        df_clean = df.copy()
        
        # 1. Standardisasi nama kolom
        column_mapping = {
            'Posting Date': 'Posting_Date',
            'Work Center': 'Work_Center', 
            'Prod Order': 'Prod_Order',
            'Confirm Qty': 'Confirm_Qty',
            'Scrab Qty': 'Scrab_Qty',
            'Confirm KG': 'Confirm_KG',
            'Act Confirm KG': 'Act_Confirm_KG',
            'Scrab KG': 'Scrab_KG',
            'Scrab Description': 'Scrab_Description',
            'Stop Time': 'Stop_Time',
            'Break Time Description': 'Break_Description',
            'Start.Date': 'Start_Date',
            'Start.Time': 'Start_Time',
            'Finis.Time': 'Finish_Time',
            'Finis.Date': 'Finish_Date'
        }
        
        df_clean = df_clean.rename(columns=column_mapping)
        
        # 2. Handle tanggal yang bermasalah
        df_clean = self._fix_date_columns(df_clean, month_name)
        
        # 3. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 4. Data type conversion
        df_clean = self._convert_data_types(df_clean)
        
        return df_clean
    
    def _fix_date_columns(self, df, month_name):
        """
        Perbaiki format tanggal yang tidak konsisten
        """
        df_fixed = df.copy()
        
        # Handle Posting_Date
        if 'Posting_Date' in df_fixed.columns:
            if month_name == 'Juni':
                # Juni menggunakan Excel serial date
                df_fixed['Posting_Date'] = pd.to_datetime(df_fixed['Posting_Date'], origin='1899-12-30', unit='D', errors='coerce')
            else:
                df_fixed['Posting_Date'] = pd.to_datetime(df_fixed['Posting_Date'], errors='coerce')
        
        # Handle Start_Date dan Finish_Date
        for date_col in ['Start_Date', 'Finish_Date']:
            if date_col in df_fixed.columns:
                if month_name == 'Juni':
                    df_fixed[date_col] = pd.to_datetime(df_fixed[date_col], origin='1899-12-30', unit='D', errors='coerce')
                else:
                    df_fixed[date_col] = pd.to_datetime(df_fixed[date_col], errors='coerce')
        
        return df_fixed
    
    def _handle_missing_values(self, df):
        """
        Handle missing values dengan strategi yang tepat
        """
        df_filled = df.copy()
        
        # Forward fill untuk Machine dan Shift (asumsi: same machine/shift continues)
        for col in ['Machine', 'Shift']:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(method='ffill')
        
        # Fill 0 untuk quantity columns
        qty_columns = ['Confirm_Qty', 'Scrab_Qty', 'Confirm_KG', 'Act_Confirm_KG', 'Scrab_KG', 'Stop_Time']
        for col in qty_columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(0)
        
        # Fill 'No Description' untuk text columns
        text_columns = ['Scrab_Description', 'Break_Description', 'Group']
        for col in text_columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna('No Description')
        
        return df_filled
    
    def _convert_data_types(self, df):
        """
        Convert ke data types yang tepat
        """
        df_converted = df.copy()
        
        # Numeric columns
        numeric_cols = ['Machine', 'Shift', 'Confirm_Qty', 'Scrab_Qty', 'Confirm_KG', 
                       'Act_Confirm_KG', 'Scrab_KG', 'Stop_Time']
        
        for col in numeric_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').fillna(0)
        
        # Categorical columns
        categorical_cols = ['Work_Center', 'Group', 'Scrab_Description', 'Break_Description']
        for col in categorical_cols:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('category')
        
        return df_converted
    
    def load_and_clean_maintenance_data(self):
        """
        Load dan bersihkan data maintenance
        """
        print("\n🔧 Loading & Cleaning Maintenance Data...")
        print("=" * 60)
        
        try:
            from pathlib import Path
            maintenance_file = Path(self.data_path) / "Flexo 104 1.xlsx"
            
            # Coba berbagai cara untuk parse maintenance data
            df = pd.read_excel(maintenance_file)
            
            # Clean header berdasarkan struktur yang ditemukan
            if len(df.columns) >= 5 and 'Unnamed' in str(df.columns[0]):
                # Header ada di baris ke-2
                new_headers = df.iloc[0].fillna('Unknown').tolist()
                df.columns = new_headers
                df = df.drop(0).reset_index(drop=True)
            
            # Remove rows yang semua NaN
            df = df.dropna(how='all')
            
            # Clean column names
            df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
            
            self.maintenance_data = df
            print(f"📊 Maintenance data loaded: {len(df)} records")
            print(f"📋 Columns: {list(df.columns)}")
            
            return self.maintenance_data
            
        except Exception as e:
            print(f"❌ Error loading maintenance data: {str(e)}")
            return None
    
    def calculate_oee_metrics(self):
        """
        Hitung metrik OEE (Overall Equipment Effectiveness)
        OEE = Availability × Performance × Quality
        """
        print("\n⚙️ Calculating OEE Metrics...")
        print("=" * 60)
        
        if self.processed_production is None:
            print("❌ No production data available")
            return None
        
        df = self.processed_production.copy()
        
        # 1. AVAILABILITY = (Planned Production Time - Downtime) / Planned Production Time
        # Asumsi: 8 jam per shift = 480 menit
        PLANNED_TIME_PER_SHIFT = 480  # minutes
        
        df['Planned_Time'] = PLANNED_TIME_PER_SHIFT
        df['Downtime'] = df['Stop_Time']
        df['Operating_Time'] = df['Planned_Time'] - df['Downtime']
        df['Availability'] = np.where(df['Planned_Time'] > 0, 
                                     df['Operating_Time'] / df['Planned_Time'], 0)
        df['Availability'] = np.clip(df['Availability'], 0, 1)  # Cap at 100%
        
        # 2. PERFORMANCE = Actual Output / Maximum Possible Output
        # Menggunakan rasio actual vs planned quantity
        df['Total_Production'] = df['Confirm_Qty'] + df['Scrab_Qty']
        
        # Calculate ideal cycle time (rata-rata production per minute)
        df['Cycle_Time'] = np.where(df['Total_Production'] > 0,
                                   df['Operating_Time'] / df['Total_Production'], 0)
        
        # Performance berdasarkan theoretical maximum
        # Asumsi: theoretical max = 2x current average performance
        avg_cycle_time = df[df['Cycle_Time'] > 0]['Cycle_Time'].median()
        theoretical_cycle_time = avg_cycle_time * 0.5 if avg_cycle_time > 0 else 1
        
        df['Theoretical_Output'] = np.where(df['Operating_Time'] > 0,
                                           df['Operating_Time'] / theoretical_cycle_time, 0)
        df['Performance'] = np.where(df['Theoretical_Output'] > 0,
                                    df['Total_Production'] / df['Theoretical_Output'], 0)
        df['Performance'] = np.clip(df['Performance'], 0, 1)  # Cap at 100%
        
        # 3. QUALITY = Good Output / Total Output
        df['Quality'] = np.where(df['Total_Production'] > 0,
                                df['Confirm_Qty'] / df['Total_Production'], 1)
        df['Quality'] = np.clip(df['Quality'], 0, 1)  # Cap at 100%
        
        # 4. OVERALL OEE
        df['OEE'] = df['Availability'] * df['Performance'] * df['Quality']
        
        # 5. Additional metrics
        df['Scrab_Rate'] = np.where(df['Total_Production'] > 0,
                                   df['Scrab_Qty'] / df['Total_Production'], 0)
        
        # Weight metrics (berat produksi)
        df['Total_Weight_KG'] = df['Act_Confirm_KG'] + df['Scrab_KG']
        df['Weight_Quality'] = np.where(df['Total_Weight_KG'] > 0,
                                       df['Act_Confirm_KG'] / df['Total_Weight_KG'], 1)
        
        self.processed_production = df
        
        # Summary statistics
        print(f"📊 OEE Metrics Summary:")
        print(f"   - Average Availability: {df['Availability'].mean():.3f} ({df['Availability'].mean()*100:.1f}%)")
        print(f"   - Average Performance: {df['Performance'].mean():.3f} ({df['Performance'].mean()*100:.1f}%)")
        print(f"   - Average Quality: {df['Quality'].mean():.3f} ({df['Quality'].mean()*100:.1f}%)")
        print(f"   - Average OEE: {df['OEE'].mean():.3f} ({df['OEE'].mean()*100:.1f}%)")
        print(f"   - Average Scrap Rate: {df['Scrab_Rate'].mean():.3f} ({df['Scrab_Rate'].mean()*100:.1f}%)")
        
        return df
    
    def create_time_features(self):
        """
        Create time-based features untuk machine learning
        """
        print("\n⏰ Creating Time-based Features...")
        print("=" * 60)
        
        if self.processed_production is None:
            print("❌ No processed data available")
            return None
        
        df = self.processed_production.copy()
        
        # Extract time features dari Posting_Date
        df['Day_of_Week'] = df['Posting_Date'].dt.dayofweek  # 0=Monday
        df['Day_of_Month'] = df['Posting_Date'].dt.day
        df['Week_of_Year'] = df['Posting_Date'].dt.isocalendar().week
        df['Month_Num'] = df['Posting_Date'].dt.month
        df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
        
        # Shift features
        df['Is_Night_Shift'] = (df['Shift'] == 3).astype(int)  # Assuming shift 3 is night
        df['Is_Day_Shift'] = (df['Shift'] == 1).astype(int)
        df['Is_Evening_Shift'] = (df['Shift'] == 2).astype(int)
        
        # Rolling window features (7-day rolling)
        df = df.sort_values('Posting_Date')
        
        rolling_features = ['OEE', 'Availability', 'Performance', 'Quality', 
                           'Downtime', 'Scrab_Rate', 'Total_Production']
        
        for feature in rolling_features:
            if feature in df.columns:
                df[f'{feature}_7d_mean'] = df[feature].rolling(window=7, min_periods=1).mean()
                df[f'{feature}_7d_std'] = df[feature].rolling(window=7, min_periods=1).std().fillna(0)
                df[f'{feature}_trend'] = df[feature] - df[f'{feature}_7d_mean']
        
        # Cumulative features
        df['Days_Since_Start'] = (df['Posting_Date'] - df['Posting_Date'].min()).dt.days
        df['Cumulative_Production'] = df['Total_Production'].cumsum()
        df['Cumulative_Downtime'] = df['Downtime'].cumsum()
        
        self.processed_production = df
        
        print(f"✅ Time features created. Total features: {len(df.columns)}")
        
        return df
    
    def create_failure_indicators(self):
        """
        Create target variables untuk klasifikasi dan regresi
        """
        print("\n🎯 Creating Failure Indicators...")
        print("=" * 60)
        
        if self.processed_production is None:
            print("❌ No processed data available")
            return None
        
        df = self.processed_production.copy()
        
        # Define failure thresholds
        OEE_THRESHOLD = 0.6  # Below 60% OEE considered poor performance
        DOWNTIME_THRESHOLD = df['Downtime'].quantile(0.8)  # Top 20% downtime
        SCRAP_THRESHOLD = df['Scrab_Rate'].quantile(0.8)  # Top 20% scrap rate
        
        # 1. CLASSIFICATION TARGETS
        # Binary failure indicators
        df['High_Downtime'] = (df['Downtime'] > DOWNTIME_THRESHOLD).astype(int)
        df['High_Scrap_Rate'] = (df['Scrab_Rate'] > SCRAP_THRESHOLD).astype(int)
        df['Poor_OEE'] = (df['OEE'] < OEE_THRESHOLD).astype(int)
        
        # Combined failure indicator
        df['Equipment_Failure'] = (
            (df['High_Downtime'] == 1) | 
            (df['High_Scrap_Rate'] == 1) | 
            (df['Poor_OEE'] == 1)
        ).astype(int)
        
        # Severity levels (0=Normal, 1=Warning, 2=Critical)
        df['Failure_Severity'] = 0
        df.loc[df['Poor_OEE'] == 1, 'Failure_Severity'] = 1
        df.loc[(df['High_Downtime'] == 1) | (df['High_Scrap_Rate'] == 1), 'Failure_Severity'] = 2
        
        # 2. REGRESSION TARGETS (RUL - Remaining Useful Life estimation)
        # Simple RUL based on performance degradation
        df['Performance_Degradation'] = df['Performance_7d_mean'].rolling(window=3).apply(
            lambda x: (x.iloc[0] - x.iloc[-1]) / x.iloc[0] if x.iloc[0] > 0 else 0
        ).fillna(0)
        
        # RUL estimation (days until next maintenance)
        # Simplified: based on degradation rate and current performance
        df['Estimated_RUL'] = np.where(
            df['Performance_Degradation'] > 0.01,  # If degrading > 1%
            np.clip((df['Performance'] - 0.5) / df['Performance_Degradation'], 1, 30),
            30  # Max 30 days
        )
        
        self.processed_production = df
        
        # Summary
        failure_rate = df['Equipment_Failure'].mean()
        print(f"📊 Failure Indicators Summary:")
        print(f"   - Equipment Failure Rate: {failure_rate:.3f} ({failure_rate*100:.1f}%)")
        print(f"   - High Downtime Events: {df['High_Downtime'].sum()}")
        print(f"   - High Scrap Rate Events: {df['High_Scrap_Rate'].sum()}")
        print(f"   - Poor OEE Events: {df['Poor_OEE'].sum()}")
        print(f"   - Average Estimated RUL: {df['Estimated_RUL'].mean():.1f} days")
        
        return df
    
    def create_visualizations(self):
        """
        Create visualizations untuk EDA
        """
        print("\n📊 Creating Visualizations...")
        print("=" * 60)
        
        if self.processed_production is None:
            print("❌ No processed data available")
            return
        
        df = self.processed_production
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('FlexoTwin Smart Maintenance - Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. OEE Trends over time
        monthly_oee = df.groupby('Month')['OEE'].mean()
        axes[0, 0].bar(monthly_oee.index, monthly_oee.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Average OEE by Month')
        axes[0, 0].set_ylabel('OEE Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Downtime distribution
        axes[0, 1].hist(df['Downtime'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Downtime Distribution')
        axes[0, 1].set_xlabel('Downtime (minutes)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Scrap Rate by Shift
        shift_scrap = df.groupby('Shift')['Scrab_Rate'].mean()
        axes[0, 2].bar(shift_scrap.index, shift_scrap.values, color='red', alpha=0.7)
        axes[0, 2].set_title('Average Scrap Rate by Shift')
        axes[0, 2].set_ylabel('Scrap Rate')
        axes[0, 2].set_xlabel('Shift')
        
        # 4. Failure indicators correlation
        failure_cols = ['High_Downtime', 'High_Scrap_Rate', 'Poor_OEE', 'Equipment_Failure']
        corr_matrix = df[failure_cols].corr()
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(failure_cols)))
        axes[1, 0].set_yticks(range(len(failure_cols)))
        axes[1, 0].set_xticklabels(failure_cols, rotation=45)
        axes[1, 0].set_yticklabels(failure_cols)
        axes[1, 0].set_title('Failure Indicators Correlation')
        
        # Add correlation values
        for i in range(len(failure_cols)):
            for j in range(len(failure_cols)):
                text = axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        # 5. Production trends
        daily_prod = df.groupby(df['Posting_Date'].dt.date)['Total_Production'].sum()
        axes[1, 1].plot(daily_prod.index, daily_prod.values, color='green', linewidth=2)
        axes[1, 1].set_title('Daily Production Trends')
        axes[1, 1].set_ylabel('Total Production')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. RUL distribution
        axes[1, 2].hist(df['Estimated_RUL'], bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Remaining Useful Life Distribution')
        axes[1, 2].set_xlabel('Estimated RUL (days)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('flexotwin_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved as 'flexotwin_analysis_dashboard.png'")
    
    def save_processed_data(self):
        """
        Save processed data untuk modeling
        """
        print("\n💾 Saving Processed Data...")
        print("=" * 60)
        
        if self.processed_production is not None:
            # Save main dataset
            self.processed_production.to_csv('flexotwin_processed_data.csv', index=False)
            print("✅ Main dataset saved: flexotwin_processed_data.csv")
            
            # Save feature-only dataset untuk modeling
            feature_columns = [col for col in self.processed_production.columns 
                             if not col.startswith('Posting_Date') and 
                             col not in ['Month', 'Year', 'Work_Center']]
            
            model_data = self.processed_production[feature_columns]
            model_data.to_csv('flexotwin_model_features.csv', index=False)
            print("✅ Model features saved: flexotwin_model_features.csv")
            
            # Save summary statistics
            summary_stats = self.processed_production.describe()
            summary_stats.to_csv('flexotwin_data_summary.csv')
            print("✅ Summary statistics saved: flexotwin_data_summary.csv")
            
            print(f"\n📈 Final Dataset Info:")
            print(f"   - Total records: {len(self.processed_production):,}")
            print(f"   - Total features: {len(self.processed_production.columns)}")
            print(f"   - Date range: {self.processed_production['Posting_Date'].min()} to {self.processed_production['Posting_Date'].max()}")
        
        if self.maintenance_data is not None:
            self.maintenance_data.to_csv('flexotwin_maintenance_data.csv', index=False)
            print("✅ Maintenance data saved: flexotwin_maintenance_data.csv")

def main():
    """
    Main function untuk menjalankan preprocessing
    """
    print("🚀 FlexoTwin Data Preprocessing & Feature Engineering")
    print("=" * 80)
    
    # Inisialisasi processor
    processor = FlexoDataProcessor("venv/Data")
    
    # 1. Load & clean data
    production_df = processor.load_and_clean_production_data()
    maintenance_df = processor.load_and_clean_maintenance_data()
    
    if production_df is not None:
        # 2. Calculate OEE metrics
        processor.calculate_oee_metrics()
        
        # 3. Create time features
        processor.create_time_features()
        
        # 4. Create failure indicators
        processor.create_failure_indicators()
        
        # 5. Create visualizations
        processor.create_visualizations()
        
        # 6. Save processed data
        processor.save_processed_data()
        
        print("\n🎉 Preprocessing completed successfully!")
        print("📋 Next steps:")
        print("   1. Review processed data and visualizations")
        print("   2. Develop classification models for failure prediction")
        print("   3. Develop regression models for RUL estimation")
        print("   4. Model evaluation and optimization")
    
    else:
        print("❌ Preprocessing failed - no production data loaded")

if __name__ == "__main__":
    main()