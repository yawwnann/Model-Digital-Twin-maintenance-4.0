import pandas as pd
import os
from datetime import datetime
import numpy as np

def analyze_excel_detailed(file_path):
    """Analisis detail struktur Excel untuk melihat semua sheet dan kolom"""
    print(f"\n{'='*50}")
    print(f"ANALISIS DETAIL: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    try:
        # Baca semua sheet
        xl_file = pd.ExcelFile(file_path)
        all_sheets = xl_file.sheet_names
        
        print(f"Total sheets: {len(all_sheets)}")
        print(f"Sheet names: {all_sheets}")
        
        flexo_data_found = {}
        
        for sheet_name in all_sheets:
            print(f"\n--- SHEET: {sheet_name} ---")
            
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                print(f"Dimensions: {df.shape}")
                
                # Cari FLEXO 104 patterns
                flexo_patterns = []
                for i in range(min(20, df.shape[0])):  # Check first 20 rows
                    for j in range(df.shape[1]):
                        cell_value = str(df.iloc[i, j]).upper()
                        if 'FLEXO' in cell_value and '104' in cell_value:
                            flexo_patterns.append((i, j, df.iloc[i, j]))
                
                if flexo_patterns:
                    print(f"FLEXO 104 found at positions: {flexo_patterns}")
                    
                    # Analisis area di sekitar FLEXO 104
                    for row_idx, col_idx, value in flexo_patterns[:1]:  # Ambil yang pertama
                        print(f"\nAnalisis area sekitar FLEXO 104 ({row_idx}, {col_idx}):")
                        
                        # Cek header/label area (beberapa baris di atas)
                        start_row = max(0, row_idx - 3)
                        end_row = min(df.shape[0], row_idx + 10)
                        start_col = max(0, col_idx - 2)
                        end_col = min(df.shape[1], col_idx + 20)
                        
                        area_df = df.iloc[start_row:end_row, start_col:end_col]
                        print("Area data:")
                        print(area_df.to_string(max_rows=10, max_cols=15))
                        
                        # Cari keywords penting
                        keywords = ['TINTA', 'INK', 'OEE', 'EFISIENSI', 'EFFICIENCY', 'PRODUKSI', 'PRODUCTION', 
                                   'KUALITAS', 'QUALITY', 'DOWNTIME', 'UPTIME', 'SPEED', 'KECEPATAN']
                        
                        found_keywords = {}
                        for k in range(df.shape[0]):
                            for l in range(df.shape[1]):
                                cell_val = str(df.iloc[k, l]).upper()
                                for keyword in keywords:
                                    if keyword in cell_val:
                                        if keyword not in found_keywords:
                                            found_keywords[keyword] = []
                                        found_keywords[keyword].append((k, l, df.iloc[k, l]))
                        
                        if found_keywords:
                            print(f"\nKeywords ditemukan:")
                            for keyword, positions in found_keywords.items():
                                print(f"  {keyword}: {positions[:3]}")  # Show first 3 occurrences
                        
                        flexo_data_found[sheet_name] = {
                            'flexo_position': (row_idx, col_idx),
                            'area_sample': area_df.to_dict(),
                            'keywords_found': found_keywords,
                            'sheet_size': df.shape
                        }
                
                # Cek apakah ada data numerik yang banyak (indikasi data produksi)
                numeric_cols = 0
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        numeric_cols += 1
                
                print(f"Numeric columns: {numeric_cols}/{len(df.columns)}")
                
                # Sample data dari sheet
                print("Sample data (first 5 rows, first 10 cols):")
                sample = df.iloc[:5, :10]
                print(sample.to_string())
                
            except Exception as e:
                print(f"Error reading sheet {sheet_name}: {str(e)}")
        
        return flexo_data_found
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return {}

def main():
    """Main function untuk analisis detail"""
    data_folder = "C:/Users/HP/Documents/Belajar python/Digital Twin/Web/Model/08_Data Produksi"
    
    excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xls')]
    excel_files.sort()
    
    print("ANALISIS DETAIL STRUKTUR EXCEL FILES")
    print("Tujuan: Memastikan semua data penting (tinta, OEE, dll) sudah tercakup")
    
    all_data = {}
    
    # Analisis hanya beberapa file untuk cek struktur
    sample_files = excel_files[:3] + excel_files[-2:]  # First 3 and last 2
    
    for file_name in sample_files:
        file_path = os.path.join(data_folder, file_name)
        flexo_data = analyze_excel_detailed(file_path)
        all_data[file_name] = flexo_data
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY ANALISIS")
    print(f"{'='*80}")
    
    all_keywords = {}
    total_sheets_with_flexo = 0
    
    for file_name, file_data in all_data.items():
        print(f"\nFile: {file_name}")
        for sheet_name, sheet_data in file_data.items():
            total_sheets_with_flexo += 1
            print(f"  Sheet: {sheet_name}")
            print(f"    FLEXO position: {sheet_data['flexo_position']}")
            print(f"    Sheet size: {sheet_data['sheet_size']}")
            
            # Compile keywords
            for keyword, positions in sheet_data['keywords_found'].items():
                if keyword not in all_keywords:
                    all_keywords[keyword] = 0
                all_keywords[keyword] += len(positions)
    
    print(f"\nTotal sheets dengan FLEXO 104: {total_sheets_with_flexo}")
    print(f"Keywords ditemukan across all files:")
    for keyword, count in sorted(all_keywords.items()):
        print(f"  {keyword}: {count} occurrences")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    critical_keywords = ['TINTA', 'INK', 'OEE', 'EFISIENSI', 'EFFICIENCY']
    missing_critical = [k for k in critical_keywords if k not in all_keywords]
    
    if missing_critical:
        print(f"⚠️  CRITICAL: Keywords tidak ditemukan: {missing_critical}")
        print("   Kemungkinan data pemakaian tinta/OEE tidak tercakup dalam ekstraksi!")
    else:
        print("✅ Semua keywords critical ditemukan")
    
    print(f"\nDari 108 records yang diekstrak:")
    print(f"- Total columns: 152")
    print(f"- Mencakup {total_sheets_with_flexo} sheets dengan FLEXO 104 data")
    
    if len(missing_critical) > 0:
        print(f"\n🔍 PERLU INVESTIGASI LEBIH LANJUT:")
        print(f"   - Cek apakah ada sheet terpisah untuk data tinta/OEE")
        print(f"   - Periksa format/nama kolom yang berbeda")
        print(f"   - Pastikan semua sheet relevan sudah diproses")

if __name__ == "__main__":
    main()