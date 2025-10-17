"""
Sensor Simulator untuk Digital Twin FLEXO Machine
==================================================

Simulator ini membaca data historis dari file Excel (XLSX) dan mensimulasikan
stream data sensor secara realtime ke API backend.

Fitur:
- Membaca data produksi, OEE, losstime dari file XLSX
- Filter data untuk mesin tertentu (default: FLEXO 104)
- Mengirim data ke Realtime OEE API via POST
- Support dry-run mode untuk testing
- Configurable interval streaming

Arsitektur:
  XLSX Files → Parser → Data Extractor → HTTP Client → API Endpoint

Usage:
  # Dry run (tanpa kirim ke API)
  python sensor_simulator.py --dry-run

  # Kirim data setiap 5 detik
  python sensor_simulator.py --interval 5

  # Spesifik mesin dan file
  python sensor_simulator.py --machine "FLEXO 104" --file "LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx"

  # Mode continuous dengan semua file
  python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx" --loop
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import requests

# Import time series generator
from time_series_generator import TimeSeriesGenerator


class FlexoSensorSimulator:
    """Simulator sensor untuk mesin FLEXO yang membaca data dari Excel."""

    def __init__(
        self,
        api_endpoint: str = "http://localhost:8100/ingest/sensor-data",
        machine_id: str = "FLEXO_104",
        dry_run: bool = False,
        verbose: bool = False,
        streaming_mode: bool = False,
        stream_interval: float = 5.0,
        simulation_interval_hours: float = 1.0
    ):
        self.api_endpoint = api_endpoint
        self.machine_id = machine_id
        self.dry_run = dry_run
        self.verbose = verbose
        self.streaming_mode = streaming_mode
        self.stream_interval = stream_interval
        self.simulation_interval_hours = simulation_interval_hours
        self.data_sent_count = 0

    def log(self, message: str, level: str = "INFO"):
        """Log messages dengan timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def debug(self, message: str):
        """Debug log (hanya jika verbose)."""
        if self.verbose:
            self.log(message, "DEBUG")

    def extract_oee_data(self, excel_path: str) -> List[Dict[str, Any]]:
        """
        Ekstrak data OEE dari file Excel.
        
        Returns list of data points dengan format:
        {
            'timestamp': '2025-09-01T08:00:00',
            'machine_id': 'FLEXO_104',
            'produced_units': 1000,
            'good_units': 950,
            'reject_units': 50,
            'runtime_minutes': 480,
            'downtime_minutes': 60,
            'design_speed': 200,
            'actual_speed': 180,
            'shift': 1,
            'oee': 0.75,
            'availability': 0.89,
            'performance': 0.90,
            'quality': 0.95
        }
        """
        self.debug(f"Membaca file: {excel_path}")
        
        try:
            xl_file = pd.ExcelFile(excel_path, engine='openpyxl')
            sheets = xl_file.sheet_names
            self.debug(f"Sheets ditemukan: {sheets}")
            
            data_points = []
            
            # 1. Ekstrak dari sheet OEE (jika ada)
            oee_sheets = [s for s in sheets if 'OEE' in s.upper() and 'WEEK' not in s.upper()]
            for sheet_name in oee_sheets:
                df = xl_file.parse(sheet_name)
                self.debug(f"Memproses sheet OEE: {sheet_name}")
                points = self._parse_oee_sheet(df, excel_path)
                data_points.extend(points)
            
            # 2. Ekstrak dari sheet produksi bulanan
            prod_sheets = [s for s in sheets if any(month in s.upper() for month in 
                          ['JANUARI', 'FEBRUARI', 'MARET', 'APRIL', 'MEI', 'JUNI',
                           'JULI', 'AGUSTUS', 'SEPTEMBER', 'OKTOBER', 'NOVEMBER', 'DESEMBER'])]
            for sheet_name in prod_sheets:
                df = xl_file.parse(sheet_name)
                self.debug(f"Memproses sheet produksi: {sheet_name}")
                points = self._parse_production_sheet(df, sheet_name)
                data_points.extend(points)
            
            # 3. Ekstrak dari sheet LOSSTIME
            losstime_sheets = [s for s in sheets if 'LOSSTIME' in s.upper()]
            losstime_data = {}
            for sheet_name in losstime_sheets:
                df = xl_file.parse(sheet_name)
                self.debug(f"Memproses sheet losstime: {sheet_name}")
                losstime_data = self._parse_losstime_sheet(df)
            
            # Merge losstime data ke data points
            if losstime_data:
                for point in data_points:
                    date_key = point.get('date')
                    if date_key in losstime_data:
                        point['downtime_minutes'] = losstime_data[date_key].get('total_minutes', 0)
                        point['losstime_reasons'] = losstime_data[date_key].get('reasons', [])
            
            self.log(f"Berhasil ekstrak {len(data_points)} data points dari {os.path.basename(excel_path)}")
            return data_points
            
        except Exception as e:
            self.log(f"Error saat membaca {excel_path}: {e}", "ERROR")
            return []

    def _parse_oee_sheet(self, df: pd.DataFrame, filepath: str) -> List[Dict[str, Any]]:
        """Parse sheet OEE untuk mendapatkan metrik OEE harian."""
        data_points = []
        
        try:
            # Cari baris yang mengandung "Design Speed" atau metrik OEE lainnya
            # Format umum: baris dengan label di kolom pertama, nilai per mesin di kolom berikutnya
            
            # Deteksi kolom FLEXO 104 atau FLEXO 4
            flexo_cols = []
            for idx, col in enumerate(df.columns):
                col_str = str(col).upper()
                if 'FLEXO' in col_str and any(x in col_str for x in ['104', '4', 'FOUR']):
                    flexo_cols.append(idx)
            
            if not flexo_cols:
                self.debug("Tidak ditemukan kolom FLEXO 104 di sheet OEE")
                return data_points
            
            # Ekstrak metrik dasar
            metrics = {}
            for idx, row in df.iterrows():
                label = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
                
                if 'design speed' in label:
                    for col_idx in flexo_cols:
                        if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                            metrics['design_speed'] = float(row.iloc[col_idx])
                
                elif 'oee' in label and 'target' not in label:
                    for col_idx in flexo_cols:
                        if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                            try:
                                oee_val = float(row.iloc[col_idx])
                                if oee_val > 1:  # Jika dalam persen
                                    oee_val = oee_val / 100
                                metrics['oee'] = oee_val
                            except:
                                pass
                
                elif 'availability' in label:
                    for col_idx in flexo_cols:
                        if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                            try:
                                val = float(row.iloc[col_idx])
                                if val > 1:
                                    val = val / 100
                                metrics['availability'] = val
                            except:
                                pass
                
                elif 'performance' in label:
                    for col_idx in flexo_cols:
                        if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                            try:
                                val = float(row.iloc[col_idx])
                                if val > 1:
                                    val = val / 100
                                metrics['performance'] = val
                            except:
                                pass
                
                elif 'quality' in label:
                    for col_idx in flexo_cols:
                        if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                            try:
                                val = float(row.iloc[col_idx])
                                if val > 1:
                                    val = val / 100
                                metrics['quality'] = val
                            except:
                                pass
            
            # Buat data point dengan metrik yang ditemukan
            if metrics:
                # Extract date from filename
                filename = os.path.basename(filepath)
                date_str = self._extract_date_from_filename(filename)
                
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'date': date_str,
                    'machine_id': self.machine_id,
                    'source': 'oee_sheet',
                    'design_speed': metrics.get('design_speed', 200),
                    'oee': metrics.get('oee'),
                    'availability': metrics.get('availability'),
                    'performance': metrics.get('performance'),
                    'quality': metrics.get('quality'),
                }
                data_points.append(data_point)
                self.debug(f"OEE metrics extracted: {metrics}")
        
        except Exception as e:
            self.debug(f"Error parsing OEE sheet: {e}")
        
        return data_points

    def _parse_production_sheet(self, df: pd.DataFrame, sheet_name: str) -> List[Dict[str, Any]]:
        """Parse sheet produksi bulanan (SEPTEMBER, OKTOBER, dll)."""
        data_points = []
        
        try:
            # Cari kolom tanggal (TGL)
            date_col_idx = None
            for idx, col in enumerate(df.columns):
                if str(col).upper().strip() in ['TGL', 'TANGGAL', 'DATE']:
                    date_col_idx = idx
                    break
            
            if date_col_idx is None:
                self.debug("Kolom tanggal tidak ditemukan di sheet produksi")
                return data_points
            
            # Cari section FLEXO 4 atau FLEXO 104
            flexo4_start_col = None
            for idx, col in enumerate(df.columns):
                col_str = str(col).upper()
                if 'FLEXO' in col_str and ('4' in col_str or '104' in col_str):
                    flexo4_start_col = idx
                    break
            
            if flexo4_start_col is None:
                self.debug("Section FLEXO 4/104 tidak ditemukan")
                return data_points
            
            # Parse data per baris (per tanggal)
            month_name = sheet_name.split()[0] if ' ' in sheet_name else sheet_name
            year = self._extract_year_from_sheet(sheet_name)
            
            for idx, row in df.iterrows():
                try:
                    # Skip header rows
                    date_val = row.iloc[date_col_idx]
                    if pd.isna(date_val) or not isinstance(date_val, (int, float)):
                        continue
                    
                    day = int(date_val)
                    if day < 1 or day > 31:
                        continue
                    
                    # Konstruksi timestamp
                    date_str = f"{year}-{self._month_name_to_number(month_name):02d}-{day:02d}"
                    
                    # Ekstrak data produksi dari kolom FLEXO 4
                    # Asumsi: kolom setelah FLEXO 4 berisi: produced, good, reject, runtime, dll
                    data_point = {
                        'timestamp': f"{date_str}T08:00:00",
                        'date': date_str,
                        'machine_id': self.machine_id,
                        'source': 'production_sheet',
                        'shift': 1,  # Default shift 1
                    }
                    
                    # Coba ekstrak nilai numerik dari kolom FLEXO 4
                    for offset in range(1, min(10, len(df.columns) - flexo4_start_col)):
                        col_idx = flexo4_start_col + offset
                        val = row.iloc[col_idx]
                        if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                            # Tentukan jenis data berdasarkan magnitude
                            if val < 100:  # Kemungkinan persentase atau metrik kecil
                                if 'oee' not in data_point:
                                    data_point['oee'] = val / 100 if val > 1 else val
                            elif val < 500:  # Kemungkinan speed atau time
                                if 'actual_speed' not in data_point:
                                    data_point['actual_speed'] = val
                            else:  # Kemungkinan produced units
                                if 'produced_units' not in data_point:
                                    data_point['produced_units'] = int(val)
                    
                    # Hanya tambahkan jika ada data valid
                    if len(data_point) > 5:
                        data_points.append(data_point)
                
                except Exception as e:
                    continue
        
        except Exception as e:
            self.debug(f"Error parsing production sheet: {e}")
        
        return data_points

    def _parse_losstime_sheet(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Parse sheet losstime untuk mendapatkan downtime per tanggal."""
        losstime_data = {}
        
        try:
            # Format: TANGGAL | MESIN | MENIT | LOSSTIME
            for idx, row in df.iterrows():
                try:
                    tanggal = row.get('TANGGAL', row.iloc[0] if len(row) > 0 else None)
                    mesin = row.get('MESIN', row.iloc[1] if len(row) > 1 else None)
                    menit = row.get('MENIT', row.iloc[2] if len(row) > 2 else None)
                    losstime = row.get('LOSSTIME', row.iloc[3] if len(row) > 3 else None)
                    
                    if pd.isna(tanggal) or pd.isna(mesin):
                        continue
                    
                    mesin_str = str(mesin).upper()
                    if 'FLEXO' not in mesin_str or ('104' not in mesin_str and '4' not in mesin_str):
                        continue
                    
                    date_key = str(int(tanggal)) if isinstance(tanggal, (int, float)) else str(tanggal)
                    
                    if date_key not in losstime_data:
                        losstime_data[date_key] = {
                            'total_minutes': 0,
                            'reasons': []
                        }
                    
                    if pd.notna(menit) and isinstance(menit, (int, float)):
                        losstime_data[date_key]['total_minutes'] += float(menit)
                    
                    if pd.notna(losstime):
                        losstime_data[date_key]['reasons'].append(str(losstime))
                
                except Exception as e:
                    continue
        
        except Exception as e:
            self.debug(f"Error parsing losstime: {e}")
        
        return losstime_data

    def _extract_date_from_filename(self, filename: str) -> str:
        """Ekstrak tanggal dari nama file."""
        # Format: "LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx"
        import re
        
        # Cari bulan
        months = {
            'JANUARI': '01', 'FEBRUARI': '02', 'MARET': '03', 'APRIL': '04',
            'MEI': '05', 'JUNI': '06', 'JULI': '07', 'AGUSTUS': '08',
            'SEPTEMBER': '09', 'OKTOBER': '10', 'NOVEMBER': '11', 'DESEMBER': '12'
        }
        
        year = '2025'  # Default
        month = '01'
        day = '01'
        
        for month_name, month_num in months.items():
            if month_name in filename.upper():
                month = month_num
                break
        
        # Cari tahun (4 digit)
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            year = year_match.group()
        
        return f"{year}-{month}-{day}"

    def _extract_year_from_sheet(self, sheet_name: str) -> str:
        """Ekstrak tahun dari nama sheet."""
        import re
        year_match = re.search(r'20\d{2}', sheet_name)
        if year_match:
            return year_match.group()
        return '2025'  # Default

    def _month_name_to_number(self, month_name: str) -> int:
        """Konversi nama bulan Indonesia ke angka."""
        months = {
            'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4,
            'MEI': 5, 'JUNI': 6, 'JULI': 7, 'AGUSTUS': 8,
            'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11, 'DESEMBER': 12
        }
        return months.get(month_name.upper(), 1)

    def send_to_api(self, data_point: Dict[str, Any]) -> bool:
        """Kirim data point ke API endpoint atau file."""
        if self.dry_run:
            # Show summary instead of full JSON for better readability
            timestamp = data_point.get('timestamp', 'N/A')
            overall = data_point.get('overall', {})
            oee = overall.get('oee', 0)
            avail = overall.get('availability', 0)
            perf = overall.get('performance', 0)
            qual = overall.get('quality', 0)
            produced = overall.get('produced_units', 0)
            
            # Component health summary
            components = data_point.get('components', {})
            comp_summary = []
            for comp_name, metrics in components.items():
                if 'health_score' in metrics:
                    comp_summary.append(f"{comp_name}:{metrics['health_score']:.0f}")
            
            comp_str = ", ".join(comp_summary) if comp_summary else "N/A"
            
            self.log(f"[DRY-RUN] {timestamp} | OEE:{oee:.2%} (A:{avail:.2%} P:{perf:.2%} Q:{qual:.2%}) | Produced:{produced} | Components: {comp_str}")
            
            if self.verbose:    
                self.log(f"[VERBOSE] Full data: {json.dumps(data_point, indent=2)}", "DEBUG")
            
            return True
        
        try:
            # Check if API endpoint is file-based (for Streamlit)
            if self.api_endpoint.startswith('file://'):
                file_path = self.api_endpoint.replace('file://', '')
                
                # Write to file
                with open(file_path, 'w') as f:
                    json.dump(data_point, f, indent=2)
                
                self.data_sent_count += 1
                
                # Show summary log
                timestamp = data_point.get('timestamp', 'N/A')
                overall = data_point.get('overall', {})
                oee = overall.get('oee', 0)
                avail = overall.get('availability', 0)
                perf = overall.get('performance', 0)
                qual = overall.get('quality', 0)
                produced = overall.get('produced_units', 0)
                
                self.log(f"✓ [{self.data_sent_count}] {timestamp} | OEE:{oee:.2%} (A:{avail:.2%} P:{perf:.2%} Q:{qual:.2%}) | Produced:{produced} | Written to file", "SUCCESS")
                
                return True
            
            # Otherwise, HTTP POST
            # Use new payload format (with components)
            payload = data_point
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                self.data_sent_count += 1
                
                # Show summary log
                timestamp = data_point.get('timestamp', 'N/A')
                overall = data_point.get('overall', {})
                oee = overall.get('oee', 0)
                avail = overall.get('availability', 0)
                perf = overall.get('performance', 0)
                qual = overall.get('quality', 0)
                produced = overall.get('produced_units', 0)
                
                self.log(f"✓ [{self.data_sent_count}] {timestamp} | OEE:{oee:.2%} (A:{avail:.2%} P:{perf:.2%} Q:{qual:.2%}) | Produced:{produced}", "SUCCESS")
                
                if self.verbose:
                    self.log(f"API Response: {response.text}", "DEBUG")
                
                return True
            else:
                self.log(f"✗ API returned status {response.status_code}: {response.text}", "ERROR")
                return False
        
        except requests.exceptions.RequestException as e:
            self.log(f"✗ Failed to send data: {e}", "ERROR")
            return False

    def extract_daily_data(self, excel_path: str) -> List[Dict[str, Any]]:
        """
        Ekstrak data harian (per hari) dari file Excel bulanan.
        
        Returns:
            List of daily aggregated data
        """
        self.debug(f"Extracting daily data from: {excel_path}")
        
        try:
            xl_file = pd.ExcelFile(excel_path, engine='openpyxl')
            sheets = xl_file.sheet_names
            
            daily_data = []
            
            # Cari sheet produksi bulanan
            prod_sheets = [s for s in sheets if any(month in s.upper() for month in 
                          ['JANUARI', 'FEBRUARI', 'MARET', 'APRIL', 'MEI', 'JUNI',
                           'JULI', 'AGUSTUS', 'SEPTEMBER', 'OKTOBER', 'NOVEMBER', 'DESEMBER'])]
            
            if not prod_sheets:
                self.log("Tidak ditemukan sheet produksi bulanan", "WARNING")
                return []
            
            # Ambil sheet pertama
            sheet_name = prod_sheets[0]
            self.debug(f"Membaca sheet: {sheet_name}")
            
            # Read dengan header di row 3 (0-indexed, skip 2 rows)
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, engine='openpyxl')
            
            self.debug(f"Columns found: {list(df.columns[:15])}")
            
            # Cari kolom FLEXO 104 (atau FLEXO 4)
            flexo_col = None
            for col in df.columns:
                col_str = str(col).upper().strip()
                # Check untuk FLEXO 104, FLEXO 4, FLEXO104, FLEXO4, dll
                if 'FLEXO' in col_str:
                    # Extract angka setelah FLEXO
                    import re
                    match = re.search(r'FLEXO\s*(\d+)', col_str)
                    if match:
                        flexo_num = match.group(1)
                        # FLEXO 104 atau FLEXO 4 sama saja
                        if flexo_num in ['104', '4', '10-4', '10 4']:
                            flexo_col = col
                            self.debug(f"Found FLEXO 104/4 column: {col}")
                            break
            
            if flexo_col is None:
                available_flexo = [c for c in df.columns if 'FLEXO' in str(c).upper()]
                self.log(f"Kolom FLEXO 104/4 tidak ditemukan. Available: {available_flexo}", "WARNING")
                return []
            
            # Parse data harian (skip header rows)
            for idx, row in df.iterrows():
                try:
                    # Get tanggal (biasanya di kolom pertama atau kedua)
                    tanggal = None
                    for possible_date_col in [df.columns[0], df.columns[1], 'TGL', 'TANGGAL', 'DATE']:
                        if possible_date_col in df.columns:
                            tanggal = row.get(possible_date_col)
                            if not pd.isna(tanggal):
                                break
                    
                    if pd.isna(tanggal):
                        continue
                    
                    # Extract day number
                    if isinstance(tanggal, (int, float)):
                        day = int(tanggal)
                    elif isinstance(tanggal, datetime):
                        day = tanggal.day
                    else:
                        try:
                            day = int(str(tanggal).split()[0])
                        except:
                            day = idx + 1
                    
                    # Skip jika bukan tanggal valid (1-31)
                    if day < 1 or day > 31:
                        continue
                    
                    # Extract production value
                    prod_value = row.get(flexo_col)
                    if pd.isna(prod_value):
                        prod_value = 0
                    
                    produced_units = int(prod_value) if isinstance(prod_value, (int, float)) else 0
                    
                    # Skip jika production 0
                    if produced_units == 0:
                        continue
                    
                    # Estimate metrics (akan di-refine oleh generator)
                    good_units = int(produced_units * 0.92)  # Assume 92% quality
                    runtime = 1200 + (produced_units / 200) * 60  # Rough estimate
                    downtime = 240 if produced_units < 18000 else 180
                    
                    daily_data.append({
                        'day': day,
                        'produced_units': produced_units,
                        'good_units': good_units,
                        'runtime_minutes': runtime,
                        'downtime_minutes': downtime,
                        'design_speed': 250,
                        'actual_speed': 210,
                        'oee': (runtime / (runtime + downtime)) * (210 / 250) * (good_units / produced_units if produced_units > 0 else 0),
                        'availability': runtime / (runtime + downtime),
                        'performance': 210 / 250,
                        'quality': good_units / produced_units if produced_units > 0 else 0
                    })
                    
                except Exception as e:
                    self.debug(f"Error parsing row {idx}: {e}")
                    continue
            
            self.log(f"Extracted {len(daily_data)} daily data points")
            return daily_data
            
        except Exception as e:
            self.log(f"Error extracting daily data: {e}", "ERROR")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return []

    def stream_from_file(self, excel_path: str):
        """
        Stream data dari file Excel dengan interpolasi time-series.
        
        OPSI B: Fixed Interval Simulation
        Data bulanan dipecah per interval simulasi tetap (default: 1 jam).
        
        Args:
            excel_path: Path ke file Excel
        """
        self.log(f"Streaming mode: Fixed interval simulation")
        self.log(f"File: {os.path.basename(excel_path)}")
        self.log(f"Simulation interval: {self.simulation_interval_hours} hour(s) per data point")
        self.log(f"Streaming interval: {self.stream_interval}s realtime")
        
        # Extract daily data
        daily_data = self.extract_daily_data(excel_path)
        
        if not daily_data:
            self.log("Tidak ada data untuk di-stream", "WARNING")
            return
        
        # Detect month/year from filename
        filename = os.path.basename(excel_path).upper()
        month_names = {
            'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4,
            'MEI': 5, 'JUNI': 6, 'JULI': 7, 'AGUSTUS': 8,
            'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11, 'DESEMBER': 12
        }
        
        month = 9  # Default September
        year = 2025
        
        for month_name, month_num in month_names.items():
            if month_name in filename:
                month = month_num
                break
        
        # Extract year from filename
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            year = int(year_match.group())
        
        start_date = datetime(year, month, 1, 6, 0, 0)  # Start at 6 AM shift A
        
        # Create generator
        generator = TimeSeriesGenerator(
            daily_data=daily_data,
            start_date=start_date,
            simulation_speed=1.0
        )
        
        # Generate stream dengan fixed interval
        self.log(f"Generating time-series stream...")
        data_stream = generator.generate_stream(
            interval_seconds=self.stream_interval,
            simulation_interval_hours=self.simulation_interval_hours
        )
        
        total_points = len(data_stream)
        total_duration = total_points * self.stream_interval
        simulated_hours = len(daily_data) * 24
        
        self.log(f"Generated {total_points} data points")
        self.log(f"Simulated period: ~{len(daily_data)} days = {simulated_hours} hours")
        self.log(f"Estimated streaming duration: {total_duration / 60:.1f} minutes ({total_duration / 3600:.2f} hours)")
        self.log("=" * 60)
        
        # Stream data
        for idx, data_point in enumerate(data_stream, 1):
            self.log(f"Streaming [{idx}/{total_points}] {data_point['timestamp']}")
            
            success = self.send_to_api(data_point)
            
            if success and idx < total_points:
                time.sleep(self.stream_interval)
            elif not success:
                self.log("Retrying in 5 seconds...", "WARNING")
                time.sleep(5)
        
        self.log(f"Streaming complete. Total data sent: {self.data_sent_count}")

    def simulate_from_file(self, excel_path: str, interval: float = 1.0, max_points: int = None):
        """
        Simulate sensor data dari satu file Excel.
        
        Args:
            excel_path: Path ke file Excel
            interval: Delay antar pengiriman data (detik)
            max_points: Maksimal data points yang dikirim (None = semua)
        """
        # Check streaming mode
        if self.streaming_mode:
            self.stream_from_file(excel_path)
            return
        
        # Normal mode (legacy)
        self.log(f"Memulai simulasi dari: {os.path.basename(excel_path)}")
        
        data_points = self.extract_oee_data(excel_path)
        
        if not data_points:
            self.log("Tidak ada data yang bisa diekstrak dari file ini", "WARNING")
            return
        
        self.log(f"Ditemukan {len(data_points)} data points")
        
        points_to_send = data_points[:max_points] if max_points else data_points
        
        for idx, point in enumerate(points_to_send, 1):
            self.log(f"Mengirim data point {idx}/{len(points_to_send)}...")
            self.send_to_api(point)
            
            if idx < len(points_to_send):
                time.sleep(interval)
        
        self.log(f"Simulasi selesai. Total data dikirim: {self.data_sent_count}")

    def simulate_from_folder(self, folder_path: str, interval: float = 1.0, loop: bool = False, max_files: int = None):
        """
        Simulate sensor data dari semua file dalam folder.
        
        Args:
            folder_path: Path ke folder berisi file Excel
            interval: Delay antar pengiriman data (detik)
            loop: Jika True, ulangi terus dari awal setelah selesai
            max_files: Maksimal file yang diproses (None = semua)
        """
        if not os.path.isdir(folder_path):
            self.log(f"Folder tidak ditemukan: {folder_path}", "ERROR")
            return
        
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.lower().endswith('.xlsx') and not f.startswith('~')]
        files.sort()
        
        if not files:
            self.log(f"Tidak ada file .xlsx di folder: {folder_path}", "ERROR")
            return
        
        self.log(f"Ditemukan {len(files)} file Excel")
        
        files_to_process = files[:max_files] if max_files else files
        
        iteration = 1
        while True:
            if loop:
                self.log(f"=== Iterasi #{iteration} ===")
            
            for file_path in files_to_process:
                self.simulate_from_file(file_path, interval)
            
            if not loop:
                break
            
            iteration += 1
            self.log(f"Menunggu {interval * 10} detik sebelum iterasi berikutnya...")
            time.sleep(interval * 10)
        
        self.log("Simulasi folder selesai")


def main():
    parser = argparse.ArgumentParser(
        description='Sensor Simulator untuk Digital Twin FLEXO Machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run satu file
  python sensor_simulator.py --file "data.xlsx" --dry-run

  # Kirim data dari satu file ke API
  python sensor_simulator.py --file "data.xlsx" --interval 2

  # STREAMING MODE: Breakdown data bulanan menjadi time-series per jam
  python sensor_simulator.py --file "SEPTEMBER 2025.xlsx" --stream --stream-interval 5 --sim-interval 1.0

  # Proses semua file di folder
  python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx"

  # Mode continuous loop
  python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx" --loop --interval 5
        """
    )
    
    # Input options
    parser.add_argument('--file', type=str, help='Path ke file Excel tunggal')
    parser.add_argument('--folder', type=str, help='Path ke folder berisi file Excel')
    parser.add_argument('--machine', type=str, default='FLEXO_104', help='Machine ID (default: FLEXO_104)')
    
    # API options
    parser.add_argument('--api', type=str, default='http://localhost:8100/ingest/sensor-data',
                       help='API endpoint (default: http://localhost:8100/ingest/sensor-data)')
    
    # Simulation options
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Interval antar pengiriman data dalam detik (default: 1.0)')
    parser.add_argument('--max-points', type=int, help='Maksimal data points per file')
    parser.add_argument('--max-files', type=int, help='Maksimal file yang diproses')
    parser.add_argument('--loop', action='store_true', help='Loop terus menerus')
    
    # Streaming mode options (NEW)
    parser.add_argument('--stream', action='store_true', 
                       help='Enable streaming mode: breakdown data bulanan per interval simulasi fixed')
    parser.add_argument('--stream-interval', type=float, default=5.0,
                       help='Interval streaming dalam detik real (default: 5.0)')
    parser.add_argument('--sim-interval', type=float, default=1.0,
                       help='Interval simulasi per data point dalam JAM (default: 1.0 = 720 points untuk 30 hari)')
    
    # Mode options
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (tidak kirim ke API)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Validasi input
    if not args.file and not args.folder:
        # Default ke folder data_xlsx
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_folder = os.path.join(script_dir, '..', '08_Data Produksi', 'data_xlsx')
        
        if os.path.isdir(default_folder):
            args.folder = default_folder
            print(f"Menggunakan folder default: {default_folder}")
        else:
            parser.error("Harus spesifikasi --file atau --folder")
    
    # Inisialisasi simulator
    simulator = FlexoSensorSimulator(
        api_endpoint=args.api,
        machine_id=args.machine,
        dry_run=args.dry_run,
        verbose=args.verbose,
        streaming_mode=args.stream,
        stream_interval=args.stream_interval,
        simulation_interval_hours=args.sim_interval
    )
    
    simulator.log("=" * 60)
    simulator.log("FLEXO Digital Twin - Sensor Simulator")
    simulator.log("=" * 60)
    simulator.log(f"Machine ID: {args.machine}")
    simulator.log(f"API Endpoint: {args.api}")
    simulator.log(f"Mode: {'STREAMING' if args.stream else 'NORMAL'} | {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.stream:
        simulator.log(f"Stream Interval: {args.stream_interval}s realtime")
        simulator.log(f"Simulation Interval: {args.sim_interval} hour(s) per data point")
        total_hours = 30 * 24  # Assume 30 days
        estimated_points = int(total_hours / args.sim_interval)
        estimated_duration = estimated_points * args.stream_interval
        simulator.log(f"Estimated: {estimated_points} points in {estimated_duration/60:.1f} minutes")
    else:
        simulator.log(f"Interval: {args.interval}s")
    simulator.log("=" * 60)
    
    try:
        if args.file:
            # Simulate dari satu file
            if not os.path.isfile(args.file):
                simulator.log(f"File tidak ditemukan: {args.file}", "ERROR")
                sys.exit(1)
            
            simulator.simulate_from_file(args.file, args.interval, args.max_points)
        
        elif args.folder:
            # Simulate dari folder
            simulator.simulate_from_folder(
                args.folder,
                args.interval,
                args.loop,
                args.max_files
            )
    
    except KeyboardInterrupt:
        simulator.log("\nSimulasi dihentikan oleh user", "INFO")
    except Exception as e:
        simulator.log(f"Error: {e}", "ERROR")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    simulator.log(f"Total data berhasil dikirim: {simulator.data_sent_count}")


if __name__ == '__main__':
    main()
