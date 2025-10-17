# Spesifikasi Component Health Assessment

## Digital Twin FLEXO Machine - Rule-Based (Tanpa ML)

---

## 📋 Tujuan Penilaian Health

### Objektif Utama

- **Menilai risk/umur pakai** setiap komponen dalam skala **0-100**
  - 🟢 **Excellent** (80-100): Optimal condition
  - 🟢 **Good** (60-79): Normal operation
  - 🟡 **Warning** (40-59): Needs attention
  - 🔴 **Critical** (0-39): Immediate action required

### Decision Support

- **Preventive Maintenance**: Schedule servis sebelum failure
- **Corrective Action**: Identifikasi komponen bermasalah
- **OEE Correlation**: Link health score ke Availability, Performance, Quality

### OEE Relationship

- **Availability**: Downtime events per komponen
- **Performance**: Speed degradation indicators
- **Quality**: Defect/reject rate contributors

---

## 🔧 5 Komponen Utama FLEXO 104

### 1. PRE FEEDER

#### Metrik Wajib/Minimal

| Metrik                | Unit       | Range Normal | Keterangan                                   |
| --------------------- | ---------- | ------------ | -------------------------------------------- |
| `web_tension_dev_pct` | %          | 0-5%         | Deviasi tegangan web dari setpoint           |
| `feed_stops_hour`     | count/hour | 0-1          | Jumlah berhenti karena feeding               |
| `runtime_min`         | minutes    | -            | Waktu operasi aktif                          |
| `downtime_min`        | minutes    | -            | Downtime dengan reason "feeding/roll change" |

#### Metrik Nice-to-Have

- `tension_setpoint_N` & `tension_actual_N`: Tegangan target vs aktual (Newton)
- `roll_diameter_mm`: Diameter roll saat ini
- `splice_count`: Jumlah sambungan roll per shift
- `misfeed_count` & `double_feed_count`: Error feeding

#### KPI Turunan

```
Tension Stability = |tension_actual - tension_setpoint| / tension_setpoint × 100%
Feed Reliability Index = 1 - (feed_stops_hour / threshold_max)
Component Health Score = f(tension_stability, feed_reliability, downtime_ratio)
```

#### Aturan Health Scoring

**Tension Deviation**

- < 5% → Score: 100 (Excellent)
- 5-10% → Score: 70 (Good)
- 10-15% → Score: 50 (Warning)
- > 15% → Score: 30 (Critical)

**Feed Stops per Hour**

- 0-1 → Score: 100 (Excellent)
- 1-3 → Score: 60 (Warning)
- > 3 → Score: 20 (Critical)

**Aggregate Formula**

```python
health_prefeeder = (
    tension_score * 0.5 +
    feed_reliability_score * 0.3 +
    uptime_score * 0.2
)
```

#### OEE Impact Mapping

- **Availability** ⚠️ High: Feed stops → line stops
- **Performance** ⚠️ Medium: Tension instability → speed reduction
- **Quality** ⚠️ Low: Minor wrinkle/warp issues

---

### 2. FEEDER

#### Metrik Wajib/Minimal

| Metrik                  | Unit       | Range Normal | Keterangan                   |
| ----------------------- | ---------- | ------------ | ---------------------------- |
| `double_sheet_err_hour` | count/hour | 0-2          | Error lembar ganda           |
| `vacuum_level_dev_pct`  | %          | 0-10%        | Deviasi vacuum dari setpoint |
| `runtime_min`           | minutes    | -            | Waktu operasi                |
| `downtime_min`          | minutes    | -            | Downtime reason "feeder"     |

#### Metrik Nice-to-Have

- `photoeye_miss_count`: Sensor photo tidak detect
- `belt_slip_pct`: Persentase slip belt conveyor
- `sensor_jam_count`: Sensor blocking

#### KPI Turunan

```
Feed Accuracy Score = 100 - (double_sheet_err_hour / threshold_max × 100)
Vacuum Stability = 100 - vacuum_level_dev_pct
```

#### Aturan Health Scoring

**Double-Sheet Error**

- < 2/hour → Score: 100
- 2-5/hour → Score: 55
- > 5/hour → Score: 25

**Vacuum Deviation**

- < 10% → Score: 100
- 10-20% → Score: 60
- > 20% → Score: 30

**Aggregate Formula**

```python
health_feeder = (
    feed_accuracy_score * 0.4 +
    vacuum_stability_score * 0.3 +
    uptime_score * 0.3
)
```

#### OEE Impact Mapping

- **Availability** ⚠️ High: Misfeed → emergency stop
- **Performance** ⚠️ Medium: Micro-stops untuk koreksi
- **Quality** ⚠️ Medium: Skew sheets → defects downstream

---

### 3. PRINTING (Unit 1-4 atau Agregat)

#### Metrik Wajib/Minimal

| Metrik                      | Unit     | Range Normal | Keterangan                       |
| --------------------------- | -------- | ------------ | -------------------------------- |
| `registration_error_mm`     | mm       | 0-0.2        | Rata-rata error registrasi warna |
| `registration_error_max_mm` | mm       | 0-0.5        | Maksimum error registrasi        |
| `reject_rate_print_pct`     | %        | 0-2%         | Reject karena print defects      |
| `actual_speed`              | cuts/min | -            | Speed aktual vs design           |
| `design_speed`              | cuts/min | -            | Speed target                     |

#### Metrik Nice-to-Have

- `ink_viscosity_s`: Viscosity tinta (detik/cup)
- `ink_viscosity_dev_pct`: Deviasi dari setpoint
- `anilox_pressure_bar`: Tekanan anilox roller
- `doctor_blade_wear_pct`: Estimasi wear doctor blade
- `cylinder_temp_c`: Suhu silinder cetak
- `color_deltaE`: Color difference ΔE

#### KPI Turunan

```
Registration Quality Index = 100 - (registration_error_mm / threshold_max × 100)
Ink Stability Index = 100 - ink_viscosity_dev_pct
Print Quality Score = 100 - reject_rate_print_pct × 20
Performance Ratio = actual_speed / design_speed × 100
```

#### Aturan Health Scoring

**Registration Error**

- < 0.2 mm → Score: 100
- 0.2-0.5 mm → Score: 60
- > 0.5 mm → Score: 25

**Print Reject Rate**

- < 2% → Score: 100
- 2-5% → Score: 55
- > 5% → Score: 20

**Ink Viscosity Deviation**

- < 10% → Score: 100
- 10-20% → Score: 65
- > 20% → Score: 35

**Aggregate Formula**

```python
health_printing = (
    registration_score * 0.35 +
    print_quality_score * 0.35 +
    ink_stability_score * 0.15 +
    performance_ratio_score * 0.15
)
```

#### OEE Impact Mapping

- **Availability** ⚠️ Medium: Plate/ink change setup
- **Performance** ⚠️ High: Speed drop saat instabilitas
- **Quality** ⚠️ High: Print defects = reject langsung

---

### 4. SLOTTER (Die-Cutting)

#### Metrik Wajib/Minimal

| Metrik                | Unit    | Range Normal | Keterangan                   |
| --------------------- | ------- | ------------ | ---------------------------- |
| `miscut_pct`          | %       | 0-1%         | Dimensi potong salah         |
| `burr_height_mm`      | mm      | 0-0.1        | Tinggi burr/serpihan         |
| `blade_life_used_pct` | %       | 0-70%        | Estimasi umur pisau terpakai |
| `downtime_min`        | minutes | -            | Blade change/setup time      |

#### Metrik Nice-to-Have

- `slot_depth_variance_mm`: Variance kedalaman slot
- `anvil_wear_index`: Index keausan anvil
- `cutting_force_n`: Gaya potong aktual

#### KPI Turunan

```
Cutting Quality Index = 100 - (miscut_pct × 50 + burr_mm × 200)
Blade Health = 100 - blade_life_used_pct
Tool Readiness Score = f(blade_health, cutting_quality)
```

#### Aturan Health Scoring

**Mis-Cut Rate**

- < 1% → Score: 100
- 1-3% → Score: 55
- > 3% → Score: 20

**Burr Height**

- < 0.1 mm → Score: 100
- 0.1-0.2 mm → Score: 60
- > 0.2 mm → Score: 25

**Blade Life Used**

- < 70% → Score: 100
- 70-90% → Score: 50
- > 90% → Score: 20

**Aggregate Formula**

```python
health_slotter = (
    cutting_quality_score * 0.4 +
    blade_health_score * 0.3 +
    burr_score * 0.2 +
    uptime_score * 0.1
)
```

#### OEE Impact Mapping

- **Availability** ⚠️ Medium: Blade change downtime
- **Performance** ⚠️ Low: Slowdown saat blade tumpul
- **Quality** ⚠️ High: Dimensi salah = reject

---

### 5. DOWN STACKER (Discharge)

#### Metrik Wajib/Minimal

| Metrik            | Unit       | Range Normal | Keterangan                  |
| ----------------- | ---------- | ------------ | --------------------------- |
| `jam_count_hour`  | count/hour | 0-1          | Frekuensi macet             |
| `misstack_pct`    | %          | 0-1%         | Tumpukan tidak rapi         |
| `discharge_speed` | units/min  | -            | Speed discharge             |
| `line_speed`      | cuts/min   | -            | Speed line (sync reference) |

#### Metrik Nice-to-Have

- `sensor_false_trigger_count`: Sensor error
- `pneumatic_pressure_dev_pct`: Deviasi tekanan pneumatik
- `stack_height_variance_mm`: Variance tinggi tumpukan

#### KPI Turunan

```
Jam-Free Index = 100 - (jam_count_hour / threshold_max × 100)
Sync Accuracy = 100 - |discharge_speed - line_speed| / line_speed × 100
Stack Quality Score = 100 - misstack_pct × 50
```

#### Aturan Health Scoring

**Jam Count**

- 0-1/hour → Score: 100
- 1-3/hour → Score: 50
- > 3/hour → Score: 15

**Mis-Stack Rate**

- < 1% → Score: 100
- 1-3% → Score: 60
- > 3% → Score: 25

**Speed Sync Deviation**

- < 5% → Score: 100
- 5-10% → Score: 70
- > 10% → Score: 40

**Aggregate Formula**

```python
health_stacker = (
    jam_free_score * 0.4 +
    stack_quality_score * 0.3 +
    sync_accuracy_score * 0.2 +
    uptime_score * 0.1
)
```

#### OEE Impact Mapping

- **Availability** ⚠️ High: Jam = instant stop
- **Performance** ⚠️ Medium: Slowdown untuk stabilisasi
- **Quality** ⚠️ Low: Kerusakan karena tumpuk buruk

---

## 📦 Payload Format Standar

### Skema JSON (per tick/event)

```json
{
  "timestamp": "2025-10-17T08:15:00Z",
  "shift": "A",
  "machine_id": "FLEXO_104",

  "overall": {
    "produced_units": 1200,
    "good_units": 1150,
    "reject_units": 50,
    "runtime_minutes": 420,
    "downtime_minutes": 30,
    "design_speed_cpm": 250,
    "actual_speed_cpm": 210,
    "oee": null,
    "availability": null,
    "performance": null,
    "quality": null
  },

  "components": {
    "PRE_FEEDER": {
      "tension_dev_pct": 4.0,
      "feed_stops_hour": 0,
      "health_score": null
    },
    "FEEDER": {
      "double_sheet_hour": 1,
      "vacuum_dev_pct": 8.0,
      "health_score": null
    },
    "PRINTING": {
      "registration_error_mm": 0.18,
      "registration_error_max_mm": 0.35,
      "ink_viscosity_dev_pct": 6.0,
      "reject_rate_pct": 2.5,
      "health_score": null
    },
    "SLOTTER": {
      "miscut_pct": 0.7,
      "burr_mm": 0.06,
      "blade_life_used_pct": 60,
      "health_score": null
    },
    "DOWN_STACKER": {
      "jam_hour": 0,
      "misstack_pct": 0.5,
      "sync_dev_pct": 3.0,
      "health_score": null
    }
  },

  "events": [
    {
      "timestamp": "2025-10-17T08:10:00Z",
      "type": "downtime",
      "component": "FEEDER",
      "reason": "FEEDER UNIT TROUBLE MEKANIK",
      "duration_min": 15
    }
  ]
}
```

### Response dari API (setelah kalkulasi)

API akan menambahkan:

- `overall.oee`, `availability`, `performance`, `quality`
- `components.*.health_score` (0-100)
- `system_status`: "NORMAL" / "WARNING" / "FAULT"
- `recommendations`: Array of suggested actions

---

## 🔄 Flow Sistem

```
┌────────────────────────┐
│  Sensor Simulator      │ ← Baca data historis XLSX
│  (Python)              │   Generate component metrics
└───────────┬────────────┘
            │ POST /ingest/sensor-data
            ▼
┌────────────────────────┐
│  API Backend           │
│  (FastAPI)             │
│  • Hitung OEE otomatis │
│  • Rule-based health   │
│  • Deteksi anomali     │
└───────────┬────────────┘
            │ Store
            ▼
┌────────────────────────┐
│  SQLite / Memory       │
│  • sensor_logs         │
│  • oee_logs            │
│  • component_health    │
└───────────┬────────────┘
            │ GET /oee/latest
            │ WS /ws/oee
            ▼
┌────────────────────────┐
│  Dashboard React       │
│  • OEE realtime        │
│  • Component health    │
│  • Status: Normal/Warn │
│  • Fishbone diagram    │
└────────────────────────┘
```

---

## 🎯 Fallback Strategy

### Jika Data Terbatas (Hanya Laporan Produksi)

Bila hanya tersedia:

- `produced`, `good`, `reject`, `runtime`, `downtime`, `actual_speed`, `design_speed`

**Fallback Logic:**

1. **OEE Calculation** tetap akurat:

   ```
   Availability = runtime / (runtime + downtime)
   Performance = actual_speed / design_speed
   Quality = good / produced
   OEE = A × P × Q
   ```

2. **Component Health Estimation** (asumsi):

   - Parse `reason_code` dari LOSSTIME sheet → mapping ke komponen
   - Gunakan heuristic sederhana:
     ```python
     if "FEEDER" in reason_code:
         feeder_health -= penalty_factor
     if "PRINTING" in reason_code or reject_rate > threshold:
         printing_health -= penalty_factor
     ```

3. **Default Health Scores** jika tidak ada data spesifik:

   - Semua komponen start di 85 (Good)
   - Decrease berdasarkan:
     - Downtime frequency × severity
     - Reject rate contribution
     - Performance degradation

4. **Alert Level**:
   - OEE < 60% → FAULT
   - OEE 60-75% → WARNING
   - OEE > 75% → NORMAL

---

## 📊 Threshold Configuration

File: `config/component_thresholds.json`

```json
{
  "PRE_FEEDER": {
    "tension_dev_pct": { "good": 5, "warning": 10, "critical": 15 },
    "feed_stops_hour": { "good": 1, "warning": 3, "critical": 5 }
  },
  "FEEDER": {
    "double_sheet_hour": { "good": 2, "warning": 5, "critical": 8 },
    "vacuum_dev_pct": { "good": 10, "warning": 20, "critical": 30 }
  },
  "PRINTING": {
    "registration_error_mm": { "good": 0.2, "warning": 0.5, "critical": 1.0 },
    "reject_rate_pct": { "good": 2, "warning": 5, "critical": 10 }
  },
  "SLOTTER": {
    "miscut_pct": { "good": 1, "warning": 3, "critical": 5 },
    "burr_mm": { "good": 0.1, "warning": 0.2, "critical": 0.3 },
    "blade_life_used_pct": { "good": 70, "warning": 90, "critical": 95 }
  },
  "DOWN_STACKER": {
    "jam_hour": { "good": 1, "warning": 3, "critical": 5 },
    "misstack_pct": { "good": 1, "warning": 3, "critical": 5 }
  }
}
```

---

## 🔧 Implementation Checklist

- [x] Dokumentasi spesifikasi lengkap
- [ ] Update sensor_simulator.py untuk generate component metrics
- [ ] Buat realtime_oee_api.py dengan health calculation
- [ ] Buat component_health_calculator.py (rule engine)
- [ ] Buat config/component_thresholds.json
- [ ] Update database schema (tambah tabel component_health)
- [ ] Test dengan data historis
- [ ] Integrasi ke dashboard frontend

---

## 📝 Notes

- Threshold dapat disesuaikan berdasarkan data lapangan
- Health score formula dapat di-tune sesuai prioritas bisnis
- Sistem dirancang modular: mudah tambah komponen baru
- Fallback strategy ensures minimum viable output

---

**Version**: 1.0  
**Last Updated**: 2025-10-17  
**Owner**: Digital Twin Development Team
