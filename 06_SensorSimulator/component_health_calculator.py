"""
Component Health Calculator - Rule-Based Engine
================================================

Module untuk menghitung health score setiap komponen mesin FLEXO
berdasarkan metrik sensor dan threshold yang telah ditentukan.

Tidak menggunakan ML - pure rule-based logic.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class HealthLevel(Enum):
    """Enum untuk kategori health level."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


class SystemStatus(Enum):
    """Enum untuk status sistem overall."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    FAULT = "FAULT"


@dataclass
class ComponentHealth:
    """Data class untuk hasil health assessment komponen."""
    component_name: str
    health_score: float  # 0-100
    health_level: HealthLevel
    metrics: Dict[str, float]
    recommendations: List[str]
    oee_impact: Dict[str, str]


@dataclass
class SystemHealth:
    """Data class untuk hasil health assessment sistem."""
    overall_health_score: float
    system_status: SystemStatus
    components: List[ComponentHealth]
    oee_metrics: Dict[str, float]
    recommendations: List[str]


class ComponentHealthCalculator:
    """
    Calculator untuk menghitung health score komponen berdasarkan rules.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize calculator dengan config file.
        
        Args:
            config_path: Path ke file config JSON. Jika None, gunakan default.
        """
        if config_path is None:
            # Default path relatif ke script ini
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config', 'component_thresholds.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.components_config = self.config.get('components', {})
        self.system_config = self.config.get('system_level', {})
        self.fallback_config = self.config.get('fallback_strategy', {})
    
    def calculate_metric_score(
        self,
        value: float,
        thresholds: Dict[str, List[float]]
    ) -> Tuple[float, HealthLevel]:
        """
        Hitung score untuk satu metrik berdasarkan threshold.
        
        Args:
            value: Nilai metrik aktual
            thresholds: Dict threshold {level: [min, max]}
        
        Returns:
            Tuple (score, health_level)
        """
        # Score mapping
        score_map = {
            'excellent': 100,
            'good': 75,
            'warning': 50,
            'critical': 25
        }
        
        # Cek setiap threshold level
        for level in ['excellent', 'good', 'warning', 'critical']:
            range_min, range_max = thresholds.get(level, [0, 0])
            if range_min <= value <= range_max:
                return score_map[level], HealthLevel(level)
        
        # Default jika di luar range
        return 25, HealthLevel.CRITICAL
    
    def calculate_component_health(
        self,
        component_name: str,
        metrics: Dict[str, float]
    ) -> ComponentHealth:
        """
        Hitung health score untuk satu komponen.
        
        Args:
            component_name: Nama komponen (e.g., "PRE_FEEDER")
            metrics: Dict metrik {metric_name: value}
        
        Returns:
            ComponentHealth object
        """
        comp_config = self.components_config.get(component_name, {})
        metrics_config = comp_config.get('metrics', {})
        
        if not metrics_config:
            # Fallback jika config tidak ada
            return self._fallback_component_health(component_name, metrics)
        
        weighted_score = 0.0
        total_weight = 0.0
        worst_level = HealthLevel.EXCELLENT
        recommendations = []
        
        # Hitung score per metrik
        for metric_name, metric_config in metrics_config.items():
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            thresholds = metric_config.get('thresholds', {})
            weight = metric_config.get('weight', 0.0)
            
            score, level = self.calculate_metric_score(value, thresholds)
            weighted_score += score * weight
            total_weight += weight
            
            # Track worst level
            level_priority = {
                HealthLevel.EXCELLENT: 0,
                HealthLevel.GOOD: 1,
                HealthLevel.WARNING: 2,
                HealthLevel.CRITICAL: 3
            }
            if level_priority[level] > level_priority[worst_level]:
                worst_level = level
            
            # Generate recommendations
            if level in [HealthLevel.WARNING, HealthLevel.CRITICAL]:
                recommendations.append(
                    f"{metric_name}: {value:.2f} {metric_config.get('unit', '')} "
                    f"- Level {level.value.upper()}"
                )
        
        # Final score
        final_score = weighted_score / total_weight if total_weight > 0 else 50
        
        # OEE impact
        oee_impact = comp_config.get('oee_impact', {})
        
        # Generate general recommendations
        if worst_level == HealthLevel.CRITICAL:
            recommendations.insert(0, f"⚠️ CRITICAL: {component_name} needs immediate attention!")
        elif worst_level == HealthLevel.WARNING:
            recommendations.insert(0, f"⚠️ WARNING: {component_name} requires monitoring")
        
        return ComponentHealth(
            component_name=component_name,
            health_score=round(final_score, 2),
            health_level=worst_level,
            metrics=metrics,
            recommendations=recommendations,
            oee_impact=oee_impact
        )
    
    def _fallback_component_health(
        self,
        component_name: str,
        metrics: Dict[str, float]
    ) -> ComponentHealth:
        """
        Fallback calculation jika config tidak tersedia.
        
        Gunakan heuristic sederhana berdasarkan uptime ratio saja.
        """
        default_score = self.fallback_config.get('default_health_score', 85)
        
        uptime_ratio = metrics.get('uptime_ratio', 1.0)
        
        # Simple scoring
        if uptime_ratio >= 0.95:
            score = 100
            level = HealthLevel.EXCELLENT
        elif uptime_ratio >= 0.85:
            score = 75
            level = HealthLevel.GOOD
        elif uptime_ratio >= 0.70:
            score = 50
            level = HealthLevel.WARNING
        else:
            score = 25
            level = HealthLevel.CRITICAL
        
        return ComponentHealth(
            component_name=component_name,
            health_score=score,
            health_level=level,
            metrics=metrics,
            recommendations=[f"Fallback calculation: uptime={uptime_ratio:.2%}"],
            oee_impact={}
        )
    
    def calculate_system_health(
        self,
        components_data: Dict[str, Dict[str, float]],
        oee_metrics: Dict[str, float]
    ) -> SystemHealth:
        """
        Hitung health score sistem keseluruhan.
        
        Args:
            components_data: Dict {component_name: {metric: value}}
            oee_metrics: Dict OEE metrics (oee, availability, performance, quality)
        
        Returns:
            SystemHealth object
        """
        # Calculate health per komponen
        component_healths = []
        for comp_name, metrics in components_data.items():
            health = self.calculate_component_health(comp_name, metrics)
            component_healths.append(health)
        
        # Overall health score (average)
        if component_healths:
            overall_score = sum(ch.health_score for ch in component_healths) / len(component_healths)
        else:
            overall_score = 50.0
        
        # Determine system status
        oee = oee_metrics.get('oee', 0.0)
        system_status = self._determine_system_status(oee, component_healths)
        
        # System-level recommendations
        recommendations = self._generate_system_recommendations(
            system_status,
            component_healths,
            oee_metrics
        )
        
        return SystemHealth(
            overall_health_score=round(overall_score, 2),
            system_status=system_status,
            components=component_healths,
            oee_metrics=oee_metrics,
            recommendations=recommendations
        )
    
    def _determine_system_status(
        self,
        oee: float,
        component_healths: List[ComponentHealth]
    ) -> SystemStatus:
        """Tentukan status sistem berdasarkan OEE dan component health."""
        
        # Cek apakah ada komponen critical
        has_critical = any(
            ch.health_level == HealthLevel.CRITICAL
            for ch in component_healths
        )
        
        # Cek apakah ada komponen warning
        has_warning = any(
            ch.health_level == HealthLevel.WARNING
            for ch in component_healths
        )
        
        # Cek min component health
        min_health = min(
            (ch.health_score for ch in component_healths),
            default=100
        )
        
        # Decision logic
        if oee < 0.60 or has_critical or min_health < 40:
            return SystemStatus.FAULT
        elif oee < 0.75 or has_warning or min_health < 60:
            return SystemStatus.WARNING
        else:
            return SystemStatus.NORMAL
    
    def _generate_system_recommendations(
        self,
        status: SystemStatus,
        component_healths: List[ComponentHealth],
        oee_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate rekomendasi aksi untuk sistem."""
        
        recommendations = []
        
        # Status-based recommendations
        if status == SystemStatus.FAULT:
            recommendations.append("🔴 SYSTEM FAULT: Immediate intervention required")
            recommendations.append("Stop production and investigate critical components")
        elif status == SystemStatus.WARNING:
            recommendations.append("🟡 SYSTEM WARNING: Schedule preventive maintenance")
        else:
            recommendations.append("🟢 SYSTEM NORMAL: Continue monitoring")
        
        # OEE-based recommendations
        oee = oee_metrics.get('oee', 0.0)
        availability = oee_metrics.get('availability', 0.0)
        performance = oee_metrics.get('performance', 0.0)
        quality = oee_metrics.get('quality', 0.0)
        
        if availability < 0.85:
            recommendations.append(f"⚠️ Low Availability ({availability:.1%}): Reduce downtime")
        
        if performance < 0.85:
            recommendations.append(f"⚠️ Low Performance ({performance:.1%}): Check speed settings")
        
        if quality < 0.95:
            recommendations.append(f"⚠️ Low Quality ({quality:.1%}): Reduce reject rate")
        
        # Component-specific recommendations
        critical_components = [
            ch for ch in component_healths
            if ch.health_level in [HealthLevel.CRITICAL, HealthLevel.WARNING]
        ]
        
        if critical_components:
            recommendations.append("Priority components requiring attention:")
            for comp in sorted(critical_components, key=lambda x: x.health_score):
                recommendations.append(
                    f"  - {comp.component_name}: {comp.health_score:.0f}/100 "
                    f"({comp.health_level.value.upper()})"
                )
        
        return recommendations
    
    def apply_downtime_penalty(
        self,
        component_name: str,
        current_health: float,
        downtime_minutes: float,
        total_minutes: float
    ) -> float:
        """
        Apply penalty ke health score berdasarkan downtime.
        
        Digunakan untuk fallback strategy.
        """
        if total_minutes == 0:
            return current_health
        
        downtime_ratio = downtime_minutes / total_minutes
        penalty_factor = self.fallback_config.get('downtime_penalty', {}).get(
            component_name,
            20
        )
        
        penalty = downtime_ratio * penalty_factor
        new_health = max(0, current_health - penalty)
        
        return new_health
    
    def map_reason_to_component(self, reason_code: str) -> Optional[str]:
        """
        Map reason code dari losstime ke nama komponen.
        
        Args:
            reason_code: String reason (e.g., "FEEDER UNIT TROUBLE")
        
        Returns:
            Nama komponen atau None
        """
        reason_mapping = self.config.get('reason_code_mapping', {})
        
        reason_upper = reason_code.upper()
        
        for key, component in reason_mapping.items():
            if key.upper() in reason_upper:
                return component
        
        return None


# Helper functions untuk integrasi dengan API

def calculate_oee_metrics(
    produced: int,
    good: int,
    reject: int,
    runtime_min: float,
    downtime_min: float,
    design_speed: float,
    actual_speed: float
) -> Dict[str, float]:
    """
    Calculate OEE metrics dari data dasar.
    
    Returns:
        Dict dengan keys: oee, availability, performance, quality
    """
    total_time = runtime_min + downtime_min
    
    if total_time == 0:
        availability = 0.0
    else:
        availability = runtime_min / total_time
    
    if design_speed == 0:
        performance = 0.0
    else:
        performance = actual_speed / design_speed
    
    if produced == 0:
        quality = 0.0
    else:
        quality = good / produced
    
    oee = availability * performance * quality
    
    return {
        'oee': oee,
        'availability': availability,
        'performance': performance,
        'quality': quality
    }


def format_health_response(system_health: SystemHealth) -> Dict[str, Any]:
    """
    Format SystemHealth object ke JSON-serializable dict.
    
    Untuk response API.
    """
    return {
        'overall_health_score': system_health.overall_health_score,
        'system_status': system_health.system_status.value,
        'oee_metrics': system_health.oee_metrics,
        'components': [
            {
                'name': ch.component_name,
                'health_score': ch.health_score,
                'health_level': ch.health_level.value,
                'metrics': ch.metrics,
                'recommendations': ch.recommendations,
                'oee_impact': ch.oee_impact
            }
            for ch in system_health.components
        ],
        'recommendations': system_health.recommendations
    }


# Example usage
if __name__ == '__main__':
    # Test calculator
    calculator = ComponentHealthCalculator()
    
    # Sample data
    components_data = {
        'PRE_FEEDER': {
            'tension_dev_pct': 6.5,
            'feed_stops_hour': 2,
            'uptime_ratio': 0.92
        },
        'FEEDER': {
            'double_sheet_hour': 3,
            'vacuum_dev_pct': 15.0,
            'uptime_ratio': 0.88
        },
        'PRINTING': {
            'registration_error_mm': 0.25,
            'reject_rate_pct': 3.5,
            'ink_viscosity_dev_pct': 12.0,
            'performance_ratio': 0.84
        },
        'SLOTTER': {
            'miscut_pct': 1.2,
            'burr_mm': 0.12,
            'blade_life_used_pct': 75,
            'uptime_ratio': 0.95
        },
        'DOWN_STACKER': {
            'jam_hour': 1,
            'misstack_pct': 0.8,
            'sync_dev_pct': 4.0,
            'uptime_ratio': 0.96
        }
    }
    
    oee_metrics = calculate_oee_metrics(
        produced=1200,
        good=1100,
        reject=100,
        runtime_min=420,
        downtime_min=60,
        design_speed=250,
        actual_speed=210
    )
    
    # Calculate system health
    system_health = calculator.calculate_system_health(components_data, oee_metrics)
    
    # Print hasil
    print("=" * 60)
    print("FLEXO 104 - System Health Assessment")
    print("=" * 60)
    print(f"Overall Health Score: {system_health.overall_health_score}/100")
    print(f"System Status: {system_health.system_status.value}")
    print(f"OEE: {system_health.oee_metrics['oee']:.1%}")
    print("\nComponent Health:")
    for comp in system_health.components:
        print(f"  {comp.component_name}: {comp.health_score}/100 ({comp.health_level.value})")
    
    print("\nRecommendations:")
    for rec in system_health.recommendations:
        print(f"  {rec}")
