"""
Time-Series Data Generator
===========================

Module untuk mengkonversi data bulanan (per hari) menjadi stream data
realtime dengan interval configurable (detik/menit).

Strategi:
1. Ekstrak data harian dari Excel (30 hari September)
2. Interpolasi linear antar hari untuk smooth transition
3. Generate synthetic component metrics berdasarkan OEE trend
4. Stream dengan interval konfigurabel (misal: 5 detik = 1 jam simulasi)

Contoh:
    30 hari × 24 jam = 720 jam data
    Dengan interval 5 detik → simulasi 1 bulan dalam 1 jam realtime
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np


class TimeSeriesGenerator:
    """
    Generator untuk create time-series data dari data bulanan.
    """
    
    def __init__(
        self,
        daily_data: List[Dict[str, Any]],
        start_date: datetime,
        simulation_speed: float = 1.0
    ):
        """
        Initialize generator.
        
        Args:
            daily_data: List of daily data points
            start_date: Tanggal mulai simulasi
            simulation_speed: Speed multiplier (1.0 = realtime, 10.0 = 10x faster)
        """
        self.daily_data = sorted(daily_data, key=lambda x: x.get('day', 0))
        self.start_date = start_date
        self.simulation_speed = simulation_speed
        
        # Calculate time intervals
        self.days_count = len(daily_data)
        self.hours_per_day = 24
        self.total_hours = self.days_count * self.hours_per_day
    
    def interpolate_value(
        self,
        day_idx: int,
        hour_in_day: float,
        metric_name: str
    ) -> float:
        """
        Interpolasi nilai metrik berdasarkan posisi waktu.
        
        Args:
            day_idx: Index hari (0-based)
            hour_in_day: Jam dalam hari (0-23.99)
            metric_name: Nama metrik untuk diinterpolasi
        
        Returns:
            Nilai terinterpolasi
        """
        if day_idx >= len(self.daily_data):
            day_idx = len(self.daily_data) - 1
        
        current_day = self.daily_data[day_idx]
        current_value = current_day.get(metric_name, 0)
        
        # Jika ini hari terakhir, return nilai hari ini
        if day_idx >= len(self.daily_data) - 1:
            return current_value
        
        next_day = self.daily_data[day_idx + 1]
        next_value = next_day.get(metric_name, 0)
        
        # Linear interpolation
        progress = hour_in_day / 24.0  # 0.0 - 1.0
        interpolated = current_value + (next_value - current_value) * progress
        
        return interpolated
    
    def add_realistic_noise(
        self,
        base_value: float,
        noise_factor: float = 0.05
    ) -> float:
        """
        Tambahkan noise realistis ke nilai untuk variasi natural.
        
        Args:
            base_value: Nilai dasar
            noise_factor: Faktor noise (0.05 = ±5%)
        
        Returns:
            Nilai dengan noise
        """
        noise = random.gauss(0, noise_factor * base_value)
        return max(0, base_value + noise)
    
    def generate_component_metrics(
        self,
        oee: float,
        availability: float,
        performance: float,
        quality: float,
        timestamp: datetime
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate synthetic component metrics berdasarkan OEE trend.
        
        Strategy:
        - Low availability → PRE_FEEDER/FEEDER issues
        - Low performance → PRINTING speed issues
        - Low quality → PRINTING/SLOTTER defects
        
        Args:
            oee, availability, performance, quality: OEE metrics
            timestamp: Current timestamp
        
        Returns:
            Dict component metrics
        """
        # Base health scores (inverse of problems)
        base_health = oee * 100  # 0-100
        
        # PRE_FEEDER: Affected by availability
        tension_dev = self.add_realistic_noise(
            (1 - availability) * 15,  # Low availability → high tension dev
            noise_factor=0.3
        )
        feed_stops = max(0, int(self.add_realistic_noise(
            (1 - availability) * 5,
            noise_factor=0.5
        )))
        
        # FEEDER: Also affected by availability
        double_sheet = max(0, int(self.add_realistic_noise(
            (1 - availability) * 4,
            noise_factor=0.4
        )))
        vacuum_dev = self.add_realistic_noise(
            (1 - availability) * 18,
            noise_factor=0.3
        )
        
        # PRINTING: Affected by performance and quality
        reg_error = self.add_realistic_noise(
            (1 - performance) * 0.6 + (1 - quality) * 0.3,
            noise_factor=0.2
        )
        ink_visc_dev = self.add_realistic_noise(
            (1 - performance) * 15,
            noise_factor=0.25
        )
        reject_rate = self.add_realistic_noise(
            (1 - quality) * 8,  # Quality directly affects reject
            noise_factor=0.3
        )
        
        # SLOTTER: Affected by quality
        miscut = self.add_realistic_noise(
            (1 - quality) * 4,
            noise_factor=0.3
        )
        burr = self.add_realistic_noise(
            (1 - quality) * 0.25,
            noise_factor=0.2
        )
        # Blade life increases over time
        hours_in_month = (timestamp - self.start_date).total_seconds() / 3600
        blade_life = min(95, (hours_in_month / self.total_hours) * 80 + random.uniform(0, 10))
        
        # DOWN_STACKER: Generally stable, slight correlation with availability
        jam_count = max(0, int(self.add_realistic_noise(
            (1 - availability) * 2,
            noise_factor=0.6
        )))
        misstack = self.add_realistic_noise(
            (1 - quality) * 2,
            noise_factor=0.4
        )
        sync_dev = self.add_realistic_noise(
            (1 - performance) * 8,
            noise_factor=0.3
        )
        
        return {
            'PRE_FEEDER': {
                'tension_dev_pct': round(tension_dev, 2),
                'feed_stops_hour': feed_stops,
                'uptime_ratio': round(availability, 3)
            },
            'FEEDER': {
                'double_sheet_hour': double_sheet,
                'vacuum_dev_pct': round(vacuum_dev, 2),
                'uptime_ratio': round(availability, 3)
            },
            'PRINTING': {
                'registration_error_mm': round(reg_error, 3),
                'registration_error_max_mm': round(reg_error * 1.5, 3),
                'ink_viscosity_dev_pct': round(ink_visc_dev, 2),
                'reject_rate_pct': round(reject_rate, 2),
                'performance_ratio': round(performance, 3)
            },
            'SLOTTER': {
                'miscut_pct': round(miscut, 2),
                'burr_mm': round(burr, 3),
                'blade_life_used_pct': round(blade_life, 1),
                'uptime_ratio': round(availability * 0.98, 3)  # Slightly better than overall
            },
            'DOWN_STACKER': {
                'jam_hour': jam_count,
                'misstack_pct': round(misstack, 2),
                'sync_dev_pct': round(sync_dev, 2),
                'uptime_ratio': round(availability * 1.02, 3)  # Slightly better
            }
        }
    
    def generate_stream(
        self,
        interval_seconds: float = 5.0,
        simulation_interval_hours: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Generate stream of data points dengan interval fixed simulasi.
        
        OPSI B: Fixed Interval Simulation
        - Data dipecah per interval simulasi tetap (default: 1 jam)
        - 30 hari = 720 jam → 720 data points
        - Kirim setiap `interval_seconds` realtime
        - Total durasi = (jumlah jam simulasi × interval_seconds) detik
        
        Args:
            interval_seconds: Interval pengiriman data (detik realtime)
            simulation_interval_hours: Interval waktu simulasi per data point (jam)
                                      1.0 = data setiap 1 jam simulasi (720 points)
                                      0.5 = data setiap 30 menit simulasi (1440 points)
                                      2.0 = data setiap 2 jam simulasi (360 points)
        
        Returns:
            List of data points dengan timestamp
        
        Example:
            # 1 bulan (720 jam) dengan interval 1 jam simulasi:
            # 720 data points, kirim setiap 5 detik
            # Total durasi: 720 × 5 = 3600 detik = 1 jam realtime
        """
        data_points = []
        
        # Calculate berapa jam simulasi per data point
        sim_hours_per_tick = simulation_interval_hours
        
        current_hour = 0.0
        current_timestamp = self.start_date
        
        while current_hour < self.total_hours:
            # Tentukan hari dan jam
            day_idx = int(current_hour / 24)
            hour_in_day = current_hour % 24
            
            if day_idx >= len(self.daily_data):
                break
            
            # Interpolate OEE metrics
            oee = self.interpolate_value(day_idx, hour_in_day, 'oee')
            availability = self.interpolate_value(day_idx, hour_in_day, 'availability')
            performance = self.interpolate_value(day_idx, hour_in_day, 'performance')
            quality = self.interpolate_value(day_idx, hour_in_day, 'quality')
            
            # Interpolate production metrics
            produced = int(self.interpolate_value(day_idx, hour_in_day, 'produced_units'))
            good = int(self.interpolate_value(day_idx, hour_in_day, 'good_units'))
            reject = produced - good
            
            runtime = self.interpolate_value(day_idx, hour_in_day, 'runtime_minutes')
            downtime = self.interpolate_value(day_idx, hour_in_day, 'downtime_minutes')
            actual_speed = self.interpolate_value(day_idx, hour_in_day, 'actual_speed')
            design_speed = self.interpolate_value(day_idx, hour_in_day, 'design_speed')
            
            # Generate component metrics
            components = self.generate_component_metrics(
                oee, availability, performance, quality, current_timestamp
            )
            
            # Determine shift (A/B/C based on hour)
            if 6 <= hour_in_day < 14:
                shift = "A"
            elif 14 <= hour_in_day < 22:
                shift = "B"
            else:
                shift = "C"
            
            # Create data point
            data_point = {
                'timestamp': current_timestamp.isoformat(),
                'shift': shift,
                'machine_id': 'FLEXO_104',
                'overall': {
                    'produced_units': produced,
                    'good_units': good,
                    'reject_units': reject,
                    'runtime_minutes': round(runtime, 1),
                    'downtime_minutes': round(downtime, 1),
                    'design_speed_cpm': round(design_speed, 0),
                    'actual_speed_cpm': round(actual_speed, 0),
                    'oee': round(oee, 4),
                    'availability': round(availability, 4),
                    'performance': round(performance, 4),
                    'quality': round(quality, 4)
                },
                'components': components
            }
            
            data_points.append(data_point)
            
            # Advance time
            current_hour += sim_hours_per_tick
            current_timestamp += timedelta(hours=simulation_interval_hours)
        
        return data_points


def extract_daily_data_from_excel(
    df_production: Any,
    df_oee: Any,
    machine_id: str = "FLEXO_104"
) -> List[Dict[str, Any]]:
    """
    Ekstrak data harian dari DataFrame Excel untuk satu bulan.
    
    Args:
        df_production: DataFrame sheet produksi bulanan
        df_oee: DataFrame sheet OEE
        machine_id: Machine ID to filter
    
    Returns:
        List of daily data points
    """
    daily_data = []
    
    # Implementation akan disesuaikan dengan struktur Excel aktual
    # Placeholder for now
    
    return daily_data


# Example usage
if __name__ == '__main__':
    # Sample daily data untuk September 2025 (simplified)
    sample_daily_data = []
    
    for day in range(1, 31):  # 30 hari September
        # Generate sample data dengan trend
        trend = day / 30.0  # 0.0 -> 1.0
        
        # OEE cenderung meningkat sedikit over time
        base_oee = 0.65 + trend * 0.1 + random.uniform(-0.05, 0.05)
        availability = 0.85 + trend * 0.05 + random.uniform(-0.03, 0.03)
        performance = 0.80 + trend * 0.08 + random.uniform(-0.04, 0.04)
        quality = 0.92 + trend * 0.03 + random.uniform(-0.02, 0.02)
        
        sample_daily_data.append({
            'day': day,
            'oee': base_oee,
            'availability': availability,
            'performance': performance,
            'quality': quality,
            'produced_units': int(20000 + random.uniform(-2000, 2000)),
            'good_units': int(18500 + random.uniform(-1500, 1500)),
            'runtime_minutes': 1200 + random.uniform(-100, 100),
            'downtime_minutes': 240 + random.uniform(-50, 50),
            'actual_speed': 210 + random.uniform(-15, 15),
            'design_speed': 250
        })
    
    # Create generator
    generator = TimeSeriesGenerator(
        daily_data=sample_daily_data,
        start_date=datetime(2025, 9, 1, 0, 0, 0)
    )
    
    # Generate stream: 30 hari = 720 jam dengan interval 1 jam simulasi
    # 720 data points, kirim setiap 5 detik = 3600 detik = 1 jam realtime
    
    print("Generating time-series data...")
    print("=" * 60)
    
    data_stream = generator.generate_stream(
        interval_seconds=5.0,           # 5 detik real antar pengiriman
        simulation_interval_hours=1.0   # 1 jam simulasi per data point
    )
    
    total_duration_seconds = len(data_stream) * 5
    
    print(f"Generated {len(data_stream)} data points")
    print(f"Simulation interval: 1 hour per data point")
    print(f"Streaming interval: 5 seconds realtime")
    print(f"Total streaming duration: {total_duration_seconds / 60:.1f} minutes realtime")
    print(f"Simulated period: 30 days = {len(data_stream)} hours")
    print()
    
    # Show sample points
    print("Sample data points:")
    for i in [0, len(data_stream)//2, -1]:
        if i < len(data_stream):
            dp = data_stream[i]
            print(f"\n[Point {i+1}] {dp['timestamp']}")
            print(f"  OEE: {dp['overall']['oee']:.2%}")
            print(f"  Printing Health: Registration={dp['components']['PRINTING']['registration_error_mm']:.3f}mm, "
                  f"Reject={dp['components']['PRINTING']['reject_rate_pct']:.1f}%")
