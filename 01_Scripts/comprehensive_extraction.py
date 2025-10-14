import pandas as pd
import os
from datetime import datetime
import numpy as np

def extract_comprehensive_flexo_data(file_path):
    """Ekstrak semua data penting untuk FLEXO 104 dari semua sheets"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        xl_file = pd.ExcelFile(file_path)
        all_sheets = xl_file.sheet_names
        
        # Extract month/year from filename
        filename = os.path.basename(file_path)
        month_year = extract_month_year(filename)
        
        comprehensive_data = {
            'source_file': filename,
            'period': month_year,
            'production_data': [],
            'oee_data': [],
            'ink_consumption': [],
            'achievement_data': [],
            'losstime_data': []
        }
        
        # 1. PRODUCTION DATA (Main sheet)
        main_sheet_found = False
        for sheet_name in all_sheets:
            if any(month in sheet_name.upper() for month in ['JANUARI', 'FEBRUARI', 'MARET', 'APRIL', 'MEI', 'JUNI',
                                                             'JULI', 'AGUSTUS', 'SEPTEMBER', 'OKTOBER', 'NOVEMBER', 'DESEMBER']):
                print(f"📊 Processing PRODUCTION sheet: {sheet_name}")
                production_data = extract_production_data(file_path, sheet_name, month_year)
                comprehensive_data['production_data'] = production_data
                main_sheet_found = True
                break
        
        # 2. OEE DATA
        oee_sheets = [sheet for sheet in all_sheets if 'OEE' in sheet.upper() and not 'UPDATE' in sheet.upper()]
        for sheet_name in oee_sheets:
            print(f"📈 Processing OEE sheet: {sheet_name}")
            oee_data = extract_oee_data(file_path, sheet_name, month_year)
            comprehensive_data['oee_data'].extend(oee_data)
        
        # 3. INK CONSUMPTION DATA
        ink_sheets = [sheet for sheet in all_sheets if 'PEMAKAIAN TINTA' in sheet.upper()]
        for sheet_name in ink_sheets:
            print(f"🎨 Processing INK sheet: {sheet_name}")
            ink_data = extract_ink_consumption(file_path, sheet_name, month_year)
            comprehensive_data['ink_consumption'].extend(ink_data)
        
        # 4. ACHIEVEMENT DATA
        achievement_sheets = [sheet for sheet in all_sheets if 'ACHIEVED' in sheet.upper()]
        for sheet_name in achievement_sheets:
            print(f"🎯 Processing ACHIEVEMENT sheet: {sheet_name}")
            achievement_data = extract_achievement_data(file_path, sheet_name, month_year)
            comprehensive_data['achievement_data'].extend(achievement_data)
        
        # 5. LOSSTIME DATA
        losstime_sheets = [sheet for sheet in all_sheets if 'LOSSTIME' in sheet.upper()]
        for sheet_name in losstime_sheets:
            print(f"⏰ Processing LOSSTIME sheet: {sheet_name}")
            losstime_data = extract_losstime_data(file_path, sheet_name, month_year)
            comprehensive_data['losstime_data'].extend(losstime_data)
        
        return comprehensive_data
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return None

def extract_production_data(file_path, sheet_name, period):
    """Extract production data for FLEXO 104"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find FLEXO 104 patterns
        flexo_104_data = []
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                cell_value = str(df.iloc[i, j])
                if 'FLEXO' in cell_value.upper() and '104' in cell_value:
                    # Extract row data
                    row_data = df.iloc[i, :].fillna('').tolist()
                    flexo_104_data.append({
                        'row_index': i,
                        'data': row_data,
                        'period': period
                    })
        
        print(f"   ✅ Found {len(flexo_104_data)} FLEXO 104 production records")
        return flexo_104_data
        
    except Exception as e:
        print(f"   ❌ Error extracting production data: {str(e)}")
        return []

def extract_oee_data(file_path, sheet_name, period):
    """Extract OEE data for FLEXO 104"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        oee_records = []
        flexo_104_cols = []
        
        # Find FLEXO 104 columns
        for i in range(min(10, df.shape[0])):
            for j in range(df.shape[1]):
                cell_value = str(df.iloc[i, j])
                if 'FLEXO' in cell_value.upper() and ('4' in cell_value or '104' in cell_value):
                    flexo_104_cols.append(j)
        
        if flexo_104_cols:
            print(f"   📊 FLEXO 4/104 found in columns: {flexo_104_cols}")
            
            # Extract OEE metrics
            metrics = ['Design Speed', 'Calendar Time', 'Operating Time', 'Planned Downtime', 
                      'Available Time', 'Unplanned Downtime', 'Net Operating Time', 'Total Cuts',
                      'Reject Cuts', 'Good Cuts', 'Availability', 'Performance', 'Quality', 'OEE']
            
            for metric in metrics:
                for i in range(df.shape[0]):
                    cell_val = str(df.iloc[i, 0])
                    if metric.upper() in cell_val.upper():
                        for col in flexo_104_cols:
                            if col < df.shape[1]:
                                value = df.iloc[i, col]
                                if pd.notna(value) and str(value).strip() != '':
                                    oee_records.append({
                                        'metric': metric,
                                        'value': value,
                                        'column': col,
                                        'period': period
                                    })
        
        print(f"   ✅ Found {len(oee_records)} OEE records")
        return oee_records
        
    except Exception as e:
        print(f"   ❌ Error extracting OEE data: {str(e)}")
        return []

def extract_ink_consumption(file_path, sheet_name, period):
    """Extract ink consumption data"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        ink_records = []
        
        # Look for FLEXO columns
        flexo_cols = []
        for i in range(min(10, df.shape[0])):
            for j in range(df.shape[1]):
                cell_value = str(df.iloc[i, j])
                if 'FLEXO' in cell_value.upper():
                    flexo_cols.append((j, cell_value))
        
        # Look for data rows with dates and numeric values
        for i in range(df.shape[0]):
            row_has_date = False
            row_has_numeric = False
            
            # Check if row has date-like values
            for j in range(min(3, df.shape[1])):
                cell_val = str(df.iloc[i, j])
                if any(char.isdigit() for char in cell_val) and len(cell_val) <= 10:
                    row_has_date = True
            
            # Check if row has numeric values (ink consumption)
            numeric_count = 0
            for j in range(df.shape[1]):
                cell_val = df.iloc[i, j]
                if pd.notna(cell_val) and str(cell_val).replace('.', '').isdigit():
                    numeric_count += 1
            
            if row_has_date and numeric_count >= 2:  # At least 2 numeric values
                row_data = df.iloc[i, :].fillna('').tolist()
                ink_records.append({
                    'row_data': row_data,
                    'period': period,
                    'numeric_values': numeric_count
                })
        
        print(f"   ✅ Found {len(ink_records)} ink consumption records")
        return ink_records
        
    except Exception as e:
        print(f"   ❌ Error extracting ink data: {str(e)}")
        return []

def extract_achievement_data(file_path, sheet_name, period):
    """Extract achievement data (target vs actual)"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        achievement_records = []
        
        # Find FLEXO columns
        flexo_patterns = []
        for i in range(min(5, df.shape[0])):
            for j in range(df.shape[1]):
                cell_value = str(df.iloc[i, j])
                if 'FLEXO' in cell_value.upper():
                    flexo_patterns.append((i, j, cell_value))
        
        # Extract achievement data for each FLEXO
        for row_idx, col_idx, flexo_name in flexo_patterns:
            # Look for data below header
            for i in range(row_idx + 2, min(row_idx + 35, df.shape[0])):
                row_data = []
                for j in range(max(0, col_idx - 1), min(col_idx + 4, df.shape[1])):
                    row_data.append(df.iloc[i, j])
                
                # Check if row has meaningful data
                numeric_count = sum(1 for val in row_data if pd.notna(val) and str(val).replace('.', '').replace('-', '').isdigit())
                if numeric_count >= 2:
                    achievement_records.append({
                        'flexo': flexo_name,
                        'row_data': row_data,
                        'period': period
                    })
        
        print(f"   ✅ Found {len(achievement_records)} achievement records")
        return achievement_records
        
    except Exception as e:
        print(f"   ❌ Error extracting achievement data: {str(e)}")
        return []

def extract_losstime_data(file_path, sheet_name, period):
    """Extract losstime data for FLEXO 104"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find FLEXO 104 column
        flexo_104_col = None
        for i in range(min(10, df.shape[0])):
            for j in range(df.shape[1]):
                cell_value = str(df.iloc[i, j])
                if 'FLEXO' in cell_value.upper() and '104' in cell_value:
                    flexo_104_col = j
                    break
            if flexo_104_col:
                break
        
        losstime_records = []
        if flexo_104_col:
            # Extract losstime data for FLEXO 104
            for i in range(df.shape[0]):
                if pd.notna(df.iloc[i, flexo_104_col]):
                    # Get surrounding data
                    row_data = []
                    for j in range(max(0, flexo_104_col - 3), min(flexo_104_col + 4, df.shape[1])):
                        row_data.append(df.iloc[i, j])
                    
                    losstime_records.append({
                        'row_data': row_data,
                        'period': period
                    })
        
        print(f"   ✅ Found {len(losstime_records)} losstime records")
        return losstime_records
        
    except Exception as e:
        print(f"   ❌ Error extracting losstime data: {str(e)}")
        return []

def extract_month_year(filename):
    """Extract month and year from filename"""
    # Month mapping
    months = {
        'JANUARI': '01', 'FEBRUARI': '02', 'MARET': '03', 'APRIL': '04',
        'MEI': '05', 'JUNI': '06', 'JULI': '07', 'AGUSTUS': '08',
        'SEPTEMBER': '09', 'OKTOBER': '10', 'NOVEMBER': '11', 'DESEMBER': '12'
    }
    
    filename_upper = filename.upper()
    
    # Find month
    month_num = None
    for month_name, month_code in months.items():
        if month_name in filename_upper:
            month_num = month_code
            break
    
    # Find year
    year = None
    for word in filename.split():
        if word.isdigit() and len(word) == 4 and word.startswith('20'):
            year = word
            break
    
    if month_num and year:
        return f"{year}-{month_num}"
    else:
        return "unknown"

def consolidate_comprehensive_data(all_files_data):
    """Consolidate all comprehensive data into structured format"""
    consolidated = {
        'production_records': [],
        'oee_records': [],
        'ink_records': [],
        'achievement_records': [],
        'losstime_records': []
    }
    
    for file_data in all_files_data:
        if file_data:
            # Production data
            for prod_data in file_data['production_data']:
                consolidated['production_records'].append({
                    'source_file': file_data['source_file'],
                    'period': file_data['period'],
                    'row_index': prod_data['row_index'],
                    'data': prod_data['data']
                })
            
            # OEE data
            for oee_data in file_data['oee_data']:
                consolidated['oee_records'].append({
                    'source_file': file_data['source_file'],
                    'period': file_data['period'],
                    'metric': oee_data['metric'],
                    'value': oee_data['value'],
                    'column': oee_data['column']
                })
            
            # Ink data
            for ink_data in file_data['ink_consumption']:
                consolidated['ink_records'].append({
                    'source_file': file_data['source_file'],
                    'period': file_data['period'],
                    'data': ink_data['row_data'],
                    'numeric_count': ink_data['numeric_values']
                })
            
            # Achievement data
            for ach_data in file_data['achievement_data']:
                consolidated['achievement_records'].append({
                    'source_file': file_data['source_file'],
                    'period': file_data['period'],
                    'flexo': ach_data['flexo'],
                    'data': ach_data['row_data']
                })
            
            # Losstime data
            for loss_data in file_data['losstime_data']:
                consolidated['losstime_records'].append({
                    'source_file': file_data['source_file'],
                    'period': file_data['period'],
                    'data': loss_data['row_data']
                })
    
    return consolidated

def save_comprehensive_data(consolidated_data, output_dir):
    """Save all comprehensive data to separate CSV files"""
    
    # 1. Save production data
    if consolidated_data['production_records']:
        prod_df = pd.DataFrame(consolidated_data['production_records'])
        prod_file = os.path.join(output_dir, 'flexo104_comprehensive_production.csv')
        prod_df.to_csv(prod_file, index=False)
        print(f"📊 Production data saved: {len(prod_df)} records → {prod_file}")
    
    # 2. Save OEE data
    if consolidated_data['oee_records']:
        oee_df = pd.DataFrame(consolidated_data['oee_records'])
        oee_file = os.path.join(output_dir, 'flexo104_comprehensive_oee.csv')
        oee_df.to_csv(oee_file, index=False)
        print(f"📈 OEE data saved: {len(oee_df)} records → {oee_file}")
    
    # 3. Save ink consumption data
    if consolidated_data['ink_records']:
        ink_df = pd.DataFrame(consolidated_data['ink_records'])
        ink_file = os.path.join(output_dir, 'flexo104_comprehensive_ink.csv')
        ink_df.to_csv(ink_file, index=False)
        print(f"🎨 Ink data saved: {len(ink_df)} records → {ink_file}")
    
    # 4. Save achievement data
    if consolidated_data['achievement_records']:
        ach_df = pd.DataFrame(consolidated_data['achievement_records'])
        ach_file = os.path.join(output_dir, 'flexo104_comprehensive_achievement.csv')
        ach_df.to_csv(ach_file, index=False)
        print(f"🎯 Achievement data saved: {len(ach_df)} records → {ach_file}")
    
    # 5. Save losstime data
    if consolidated_data['losstime_records']:
        loss_df = pd.DataFrame(consolidated_data['losstime_records'])
        loss_file = os.path.join(output_dir, 'flexo104_comprehensive_losstime.csv')
        loss_df.to_csv(loss_file, index=False)
        print(f"⏰ Losstime data saved: {len(loss_df)} records → {loss_file}")

def main():
    """Main function untuk ekstraksi data komprehensif"""
    
    print("🔍 COMPREHENSIVE FLEXO 104 DATA EXTRACTION")
    print("Mengekstrak SEMUA data penting:")
    print("- Production data (sheet utama)")
    print("- OEE data (sheet OEE)")
    print("- Ink consumption (sheet PEMAKAIAN TINTA)")
    print("- Achievement data (sheet ACHIEVED)")
    print("- Losstime data (sheet LOSSTIME)")
    
    data_folder = "C:/Users/HP/Documents/Belajar python/Digital Twin/Web/Model/08_Data Produksi"
    output_folder = "C:/Users/HP/Documents/Belajar python/Digital Twin/Web/Model/00_Data"
    
    # Get all Excel files
    excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xls')]
    excel_files.sort()
    
    print(f"\nFound {len(excel_files)} Excel files to process")
    
    all_files_data = []
    
    # Process each file
    for file_name in excel_files:
        file_path = os.path.join(data_folder, file_name)
        comprehensive_data = extract_comprehensive_flexo_data(file_path)
        if comprehensive_data:
            all_files_data.append(comprehensive_data)
    
    # Consolidate all data
    print(f"\n{'='*80}")
    print("CONSOLIDATING ALL DATA")
    print(f"{'='*80}")
    
    consolidated = consolidate_comprehensive_data(all_files_data)
    
    # Summary
    print(f"\n📋 EXTRACTION SUMMARY:")
    print(f"   📊 Production records: {len(consolidated['production_records'])}")
    print(f"   📈 OEE records: {len(consolidated['oee_records'])}")
    print(f"   🎨 Ink consumption records: {len(consolidated['ink_records'])}")
    print(f"   🎯 Achievement records: {len(consolidated['achievement_records'])}")
    print(f"   ⏰ Losstime records: {len(consolidated['losstime_records'])}")
    
    # Save all data
    save_comprehensive_data(consolidated, output_folder)
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALISIS DATA LENGKAP")
    print(f"{'='*80}")
    
    total_records = (len(consolidated['production_records']) + 
                    len(consolidated['oee_records']) + 
                    len(consolidated['ink_records']) + 
                    len(consolidated['achievement_records']) + 
                    len(consolidated['losstime_records']))
    
    print(f"Total records sebelumnya: 108 (hanya production)")
    print(f"Total records sekarang: {total_records} (semua data)")
    print(f"Peningkatan data: {total_records - 108} records tambahan")
    
    if len(consolidated['oee_records']) == 0:
        print(f"⚠️  WARNING: Tidak ada OEE data yang diekstrak!")
    else:
        print(f"✅ OEE data berhasil diekstrak")
    
    if len(consolidated['ink_records']) == 0:
        print(f"⚠️  WARNING: Tidak ada ink consumption data yang diekstrak!")
    else:
        print(f"✅ Ink consumption data berhasil diekstrak")

if __name__ == "__main__":
    main()