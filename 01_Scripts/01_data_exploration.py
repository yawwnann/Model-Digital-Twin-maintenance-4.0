"""
FlexoTwin Smart Maintenance 4.0
Data Exploration dan Loading Script

Tujuan: Memahami struktur dan kualitas data produksi & perawatan
Dibuat untuk: Proyek Skripsi Teknik Industri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class FlexoDataExplorer:
    def __init__(self, data_path):
        """
        Inisialisasi Data Explorer untuk FlexoTwin
        
        Args:
            data_path (str): Path ke folder data
        """
        self.data_path = Path(data_path)
        self.production_data = {}
        self.maintenance_data = None
        
    def load_production_data(self):
        """
        Load semua file data produksi (Februari - Juni 2025)
        """
        print("🔄 Memuat Data Produksi...")
        print("=" * 50)
        
        # Pattern untuk file produksi
        production_files = list(self.data_path.glob("Produksi Bulan *.xlsx"))
        
        for file in production_files:
            try:
                # Extract nama bulan dari filename
                month_name = file.stem.replace("Produksi Bulan ", "").replace(" 2025", "")
                
                # Load data
                df = pd.read_excel(file)
                
                # Filter hanya untuk work center C_FL104
                if 'Work Center' in df.columns:
                    original_count = len(df)
                    df = df[df['Work Center'] == 'C_FL104']
                    filtered_count = len(df)
                    print(f"🎯 Filter C_FL104: {original_count} → {filtered_count} records")
                elif 'work center' in df.columns:
                    original_count = len(df)
                    df = df[df['work center'] == 'C_FL104']
                    filtered_count = len(df)
                    print(f"🎯 Filter C_FL104: {original_count} → {filtered_count} records")
                else:
                    print("⚠️ Kolom 'Work Center' tidak ditemukan, memuat semua data")
                
                # Tampilkan info basic
                print(f"📁 File: {file.name}")
                print(f"📊 Shape (after filter): {df.shape}")
                print(f"📋 Columns: {list(df.columns)}")
                print(f"📅 Data Types:")
                for col, dtype in df.dtypes.items():
                    print(f"   - {col}: {dtype}")
                
                # Validasi data C_FL104
                if len(df) > 0:
                    self.production_data[month_name] = df
                    print(f"✅ {month_name} berhasil dimuat ({len(df)} records C_FL104)")
                else:
                    print(f"⚠️ {month_name}: Tidak ada data untuk C_FL104")
                print("-" * 30)
                
            except Exception as e:
                print(f"❌ Error loading {file.name}: {str(e)}")
        
        print(f"🎉 Total {len(self.production_data)} file produksi berhasil dimuat\n")
        
    def load_maintenance_data(self):
        """
        Load data perawatan/maintenance
        """
        print("🔧 Memuat Data Perawatan...")
        print("=" * 50)
        
        try:
            maintenance_file = self.data_path / "Flexo 104 1.xlsx"
            
            # Coba beberapa sheet jika ada
            xl_file = pd.ExcelFile(maintenance_file)
            print(f"📋 Available sheets: {xl_file.sheet_names}")
            
            # Load sheet pertama atau semua sheet
            if len(xl_file.sheet_names) == 1:
                self.maintenance_data = pd.read_excel(maintenance_file)
            else:
                # Jika ada multiple sheets, load semua
                self.maintenance_data = {}
                for sheet in xl_file.sheet_names:
                    self.maintenance_data[sheet] = pd.read_excel(maintenance_file, sheet_name=sheet)
            
            print(f"📁 File: Flexo 104 1.xlsx")
            
            if isinstance(self.maintenance_data, dict):
                for sheet_name, df in self.maintenance_data.items():
                    print(f"📊 Sheet '{sheet_name}' Shape: {df.shape}")
                    print(f"📋 Columns: {list(df.columns)}")
            else:
                print(f"📊 Shape: {self.maintenance_data.shape}")
                print(f"📋 Columns: {list(self.maintenance_data.columns)}")
                print(f"📅 Data Types:")
                for col, dtype in self.maintenance_data.dtypes.items():
                    print(f"   - {col}: {dtype}")
            
            print("✅ Data perawatan berhasil dimuat\n")
            
        except Exception as e:
            print(f"❌ Error loading maintenance data: {str(e)}")
    
    def explore_production_patterns(self):
        """
        Analisis pola data produksi
        """
        print("📈 Analisis Pola Data Produksi")
        print("=" * 50)
        
        for month, df in self.production_data.items():
            print(f"\n📅 Bulan: {month}")
            print(f"📊 Jumlah record: {len(df)}")
            
            # Cek missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("⚠️ Missing values ditemukan:")
                for col, count in missing[missing > 0].items():
                    percentage = (count / len(df)) * 100
                    print(f"   - {col}: {count} ({percentage:.1f}%)")
            else:
                print("✅ Tidak ada missing values")
            
            # Sample data
            print(f"🔍 Sample data (3 baris pertama):")
            print(df.head(3).to_string(index=False))
            print("-" * 50)
    
    def explore_maintenance_patterns(self):
        """
        Analisis pola data perawatan
        """
        print("🔧 Analisis Pola Data Perawatan")
        print("=" * 50)
        
        if isinstance(self.maintenance_data, dict):
            for sheet_name, df in self.maintenance_data.items():
                print(f"\n📋 Sheet: {sheet_name}")
                self._analyze_maintenance_sheet(df)
        else:
            self._analyze_maintenance_sheet(self.maintenance_data)
    
    def _analyze_maintenance_sheet(self, df):
        """
        Helper function untuk analisis sheet maintenance
        """
        print(f"📊 Jumlah record: {len(df)}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ Missing values ditemukan:")
            for col, count in missing[missing > 0].items():
                percentage = (count / len(df)) * 100
                print(f"   - {col}: {count} ({percentage:.1f}%)")
        else:
            print("✅ Tidak ada missing values")
        
        # Sample data
        print(f"🔍 Sample data (3 baris pertama):")
        print(df.head(3).to_string(index=False))
        print("-" * 30)
    
    def generate_data_summary(self):
        """
        Generate summary report
        """
        print("📋 RINGKASAN DATA FLEXOTWIN")
        print("=" * 60)
        
        print(f"🏭 Data Produksi:")
        print(f"   - Periode: {len(self.production_data)} bulan (Feb-Jun 2025)")
        
        total_production_records = sum(len(df) for df in self.production_data.values())
        print(f"   - Total records: {total_production_records:,}")
        
        # Common columns across production data
        if self.production_data:
            first_df = list(self.production_data.values())[0]
            print(f"   - Kolom utama: {', '.join(first_df.columns)}")
        
        print(f"\n🔧 Data Perawatan:")
        if isinstance(self.maintenance_data, dict):
            total_maintenance_records = sum(len(df) for df in self.maintenance_data.values())
            print(f"   - Total sheets: {len(self.maintenance_data)}")
        else:
            total_maintenance_records = len(self.maintenance_data) if self.maintenance_data is not None else 0
            print(f"   - Single sheet data")
        
        print(f"   - Total records: {total_maintenance_records:,}")
        
        print(f"\n🎯 Next Steps:")
        print("   1. Data cleaning & preprocessing")
        print("   2. Feature engineering (OEE calculation)")
        print("   3. Data integration & time alignment")
        print("   4. Model development preparation")

def main():
    """
    Main function untuk menjalankan eksplorasi data
    """
    print("🚀 FlexoTwin Data Explorer Starting...")
    print("=" * 60)
    
    # Path ke data
    data_path = "venv/Data"
    
    # Inisialisasi explorer
    explorer = FlexoDataExplorer(data_path)
    
    # Load semua data
    explorer.load_production_data()
    explorer.load_maintenance_data()
    
    # Eksplorasi pola
    explorer.explore_production_patterns()
    explorer.explore_maintenance_patterns()
    
    # Generate summary
    explorer.generate_data_summary()
    
    print("\n🎉 Data exploration completed!")
    print("💡 Tip: Jalankan script ini untuk memahami struktur data Anda")

if __name__ == "__main__":
    main()