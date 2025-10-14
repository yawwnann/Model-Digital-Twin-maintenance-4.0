"""
PRODUCTION DATA ANALYZER - Focus on FLEXO 104
Analyze Excel production reports from September 2024 - August 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionDataAnalyzer:
    """Analyzer for production Excel files focusing on FLEXO 104"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.flexo104_data = []
        self.analysis_results = {}
        
    def get_production_files(self):
        """Get all Excel files in production data directory"""
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.xls') or file.endswith('.xlsx'):
                files.append(os.path.join(self.data_dir, file))
        
        # Sort files chronologically
        files.sort()
        return files
    
    def extract_month_year(self, filename):
        """Extract month and year from filename"""
        try:
            # Parse different filename formats
            if "SEPTEMBER 2024" in filename.upper():
                return "2024-09"
            elif "OKTOBER 2024" in filename.upper():
                return "2024-10"  
            elif "NOVEMBER 2024" in filename.upper():
                return "2024-11"
            elif "DESEMBER 2024" in filename.upper():
                return "2024-12"
            elif "JANUARI 2025" in filename.upper():
                return "2025-01"
            elif "FEBRUARI 2025" in filename.upper():
                return "2025-02"
            elif "MARET 2025" in filename.upper():
                return "2025-03"
            elif "APRIL 2025" in filename.upper():
                return "2025-04"
            elif "MEI 2025" in filename.upper():
                return "2025-05"
            elif "JUNI 2025" in filename.upper():
                return "2025-06"
            elif "JULI 2025" in filename.upper():
                return "2025-07"
            elif "AGUSTUS 2025" in filename.upper():
                return "2025-08"
            elif "SEPTEMBER 2025" in filename.upper():
                return "2025-09"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def analyze_excel_file(self, filepath):
        """Analyze single Excel file for FLEXO 104 data"""
        filename = os.path.basename(filepath)
        month_year = self.extract_month_year(filename)
        
        print(f"\n🔍 Analyzing: {filename}")
        print(f"📅 Period: {month_year}")
        
        try:
            # Read Excel file - try different sheets
            excel_file = pd.ExcelFile(filepath)
            sheet_names = excel_file.sheet_names
            print(f"📄 Sheets found: {sheet_names}")
            
            flexo104_found = False
            file_analysis = {
                'filename': filename,
                'period': month_year,
                'sheets': sheet_names,
                'flexo104_data': None,
                'flexo104_rows': 0,
                'columns': [],
                'sample_data': None,
                'usable': False,
                'issues': []
            }
            
            # Look for FLEXO 104 data in each sheet
            for sheet_name in sheet_names:
                try:
                    print(f"   📋 Checking sheet: {sheet_name}")
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    
                    if df.empty:
                        print(f"      ⚠️  Sheet is empty")
                        continue
                    
                    print(f"      📊 Shape: {df.shape}")
                    print(f"      📝 Columns: {list(df.columns)}")
                    
                    # Look for FLEXO 104 or FL104 in the data
                    flexo104_masks = []
                    
                    # Check different column patterns
                    for col in df.columns:
                        if df[col].dtype == 'object':  # Text columns
                            mask1 = df[col].astype(str).str.contains('FLEXO.*104|FL.*104|C_FL104', case=False, na=False)
                            mask2 = df[col].astype(str).str.contains('104', case=False, na=False)
                            
                            if mask1.any():
                                flexo104_masks.append(mask1)
                                print(f"      ✅ Found FLEXO 104 references in column '{col}'")
                            elif mask2.any():
                                flexo104_masks.append(mask2)
                                print(f"      🔍 Found '104' references in column '{col}'")
                    
                    # If found FLEXO 104 data
                    if flexo104_masks:
                        # Combine all masks
                        combined_mask = flexo104_masks[0]
                        for mask in flexo104_masks[1:]:
                            combined_mask = combined_mask | mask
                        
                        flexo104_df = df[combined_mask]
                        
                        if not flexo104_df.empty:
                            flexo104_found = True
                            file_analysis['flexo104_data'] = flexo104_df
                            file_analysis['flexo104_rows'] = len(flexo104_df)
                            file_analysis['columns'] = list(flexo104_df.columns)
                            file_analysis['sample_data'] = flexo104_df.head(3).to_dict('records')
                            file_analysis['usable'] = True
                            
                            print(f"      ✅ FLEXO 104 data found: {len(flexo104_df)} rows")
                            print(f"      📋 Sample data:")
                            for i, row in enumerate(flexo104_df.head(2).iterrows()):
                                print(f"         Row {i+1}: {dict(row[1])}")
                            
                            self.flexo104_data.append({
                                'period': month_year,
                                'filename': filename,
                                'sheet': sheet_name,
                                'data': flexo104_df
                            })
                            break
                    
                    # Also check if sheet name suggests FLEXO data
                    elif any(keyword in sheet_name.upper() for keyword in ['FLEXO', 'MESIN', 'PRODUKSI', 'FL104']):
                        print(f"      🔍 Sheet name suggests production data, analyzing content...")
                        # Show sample of the data
                        print(f"      📋 Sample data from {sheet_name}:")
                        for i, row in enumerate(df.head(3).iterrows()):
                            print(f"         Row {i+1}: {dict(row[1])}")
                        
                        if not file_analysis['sample_data']:
                            file_analysis['sample_data'] = df.head(3).to_dict('records')
                            file_analysis['columns'] = list(df.columns)
                    
                except Exception as e:
                    error_msg = f"Error reading sheet '{sheet_name}': {str(e)}"
                    print(f"      ❌ {error_msg}")
                    file_analysis['issues'].append(error_msg)
            
            if not flexo104_found:
                print(f"   ❌ No FLEXO 104 data found in any sheet")
                file_analysis['issues'].append("No FLEXO 104 data found")
            
            self.analysis_results[filename] = file_analysis
            return file_analysis
            
        except Exception as e:
            error_msg = f"Error analyzing file: {str(e)}"
            print(f"   ❌ {error_msg}")
            file_analysis['issues'].append(error_msg)
            self.analysis_results[filename] = file_analysis
            return file_analysis
    
    def analyze_all_files(self):
        """Analyze all production files"""
        print("="*80)
        print("PRODUCTION DATA ANALYSIS - FLEXO 104 FOCUS")
        print("="*80)
        print("Period: September 2024 - August 2025")
        
        files = self.get_production_files()
        print(f"\n📂 Found {len(files)} Excel files to analyze")
        
        # Filter files for target period (Sep 2024 - Aug 2025)
        target_files = []
        for filepath in files:
            filename = os.path.basename(filepath)
            month_year = self.extract_month_year(filename)
            
            # Check if file is in target period
            if month_year != "unknown":
                year, month = month_year.split("-")
                year, month = int(year), int(month)
                
                # Sep 2024 - Aug 2025
                if (year == 2024 and month >= 9) or (year == 2025 and month <= 8):
                    target_files.append(filepath)
                    print(f"✅ Target period: {filename} ({month_year})")
                else:
                    print(f"❌ Outside target: {filename} ({month_year})")
            else:
                print(f"❓ Unknown period: {filename}")
        
        print(f"\n🎯 Analyzing {len(target_files)} files in target period...")
        
        # Analyze each target file
        for filepath in target_files:
            self.analyze_excel_file(filepath)
        
        return self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate summary report of analysis"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        total_files = len(self.analysis_results)
        usable_files = sum(1 for result in self.analysis_results.values() if result['usable'])
        total_flexo104_rows = sum(len(data['data']) for data in self.flexo104_data)
        
        print(f"📊 FILES ANALYZED: {total_files}")
        print(f"✅ USABLE FILES: {usable_files}")
        print(f"📈 TOTAL FLEXO 104 RECORDS: {total_flexo104_rows}")
        
        print(f"\n📅 MONTHLY BREAKDOWN:")
        print("-" * 50)
        
        monthly_data = {}
        for data in self.flexo104_data:
            period = data['period']
            if period not in monthly_data:
                monthly_data[period] = 0
            monthly_data[period] += len(data['data'])
        
        for period in sorted(monthly_data.keys()):
            print(f"   {period}: {monthly_data[period]} records")
        
        print(f"\n📋 FILE STATUS:")
        print("-" * 50)
        
        for filename, result in self.analysis_results.items():
            status = "✅ USABLE" if result['usable'] else "❌ UNUSABLE"
            rows = result['flexo104_rows'] if result['usable'] else 0
            print(f"   {result['period']} | {status} | {rows} rows | {filename}")
            
            if result['issues']:
                for issue in result['issues']:
                    print(f"      ⚠️  {issue}")
        
        print(f"\n🎯 TRAINING DATA RECOMMENDATION:")
        print("-" * 50)
        
        if total_flexo104_rows > 0:
            print(f"✅ DATA AVAILABLE FOR TRAINING!")
            print(f"   📊 Total records: {total_flexo104_rows}")
            print(f"   📅 Period coverage: {min(monthly_data.keys())} to {max(monthly_data.keys())}")
            print(f"   📈 Monthly average: {total_flexo104_rows / len(monthly_data):.0f} records")
            
            if total_flexo104_rows >= 500:
                print(f"   🎯 EXCELLENT: Sufficient data for robust ML training")
            elif total_flexo104_rows >= 200:
                print(f"   ✅ GOOD: Adequate data for ML training")  
            else:
                print(f"   ⚠️  LIMITED: May need data augmentation for training")
        else:
            print(f"❌ NO USABLE DATA FOUND")
            print(f"   Recommendation: Check data format and FLEXO 104 naming conventions")
        
        print("="*80)
        
        return {
            'total_files': total_files,
            'usable_files': usable_files,
            'total_records': total_flexo104_rows,
            'monthly_breakdown': monthly_data,
            'file_results': self.analysis_results,
            'flexo104_data': self.flexo104_data
        }

if __name__ == "__main__":
    # Analyze production data
    data_dir = "../08_Data Produksi"
    analyzer = ProductionDataAnalyzer(data_dir)
    
    # Run analysis
    summary = analyzer.analyze_all_files()
    
    # Save results if data found
    if summary['total_records'] > 0:
        print(f"\n💾 Saving consolidated FLEXO 104 data...")
        
        all_data = []
        for data_entry in analyzer.flexo104_data:
            df = data_entry['data'].copy()
            df['source_file'] = data_entry['filename']
            df['period'] = data_entry['period']
            all_data.append(df)
        
        if all_data:
            consolidated_df = pd.concat(all_data, ignore_index=True)
            output_file = '../00_Data/flexo104_production_real_consolidated.csv'
            consolidated_df.to_csv(output_file, index=False)
            print(f"✅ Consolidated data saved: {output_file}")
            print(f"📊 Final dataset: {len(consolidated_df)} rows, {len(consolidated_df.columns)} columns")