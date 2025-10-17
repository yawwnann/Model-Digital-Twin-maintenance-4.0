# 🎬 Streaming Mode Guide

## Overview

**Streaming Mode** adalah fitur untuk mengkonversi data bulanan (30 hari) menjadi stream realtime dengan interval konfigurabel. Mode ini sangat cocok untuk:

- ✅ Simulasi kondisi produksi realtime dari data historis
- ✅ Testing API backend dengan data berkelanjutan
- ✅ Demo dashboard dengan data yang "hidup"
- ✅ Development tanpa perlu sensor fisik

## Cara Kerja

```
Data Excel (30 hari) → Ekstrak Harian → Interpolasi → Time-Series Stream
     SEPTEMBER.xlsx       20-30 points      Linear       120-1200 points
                                                         (per detik/menit)
```

### Proses Detail:

1. **Ekstraksi Harian**: Baca data produksi per hari dari Excel (kolom FLEXO 4)
2. **Interpolasi Linear**: Smooth transition antar hari
3. **Component Metrics Generation**: Generate metrik per komponen berdasarkan OEE trend
4. **Streaming**: Kirim data dengan interval configurable

## Quick Examples

### Example 1: Demo Dashboard (10 menit)

**Goal**: Stream data 1 bulan dalam 10 menit untuk demo

```powershell
python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream `
  --stream-interval 5 `
  --stream-speed 4320 `
  --api "http://localhost:8100/ingest/sensor-data"
```

**Output**: ~120 data points, 1 point setiap 5 detik

### Example 2: Development Testing (1 jam)

**Goal**: Stream lebih lambat untuk testing detail

```powershell
python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream `
  --stream-interval 3 `
  --stream-speed 2592 `
  --verbose
```

**Output**: ~1200 data points, 1 point setiap 3 detik

### Example 3: Quick Test (30 detik)

**Goal**: Test cepat end-to-end

```powershell
python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream `
  --stream-interval 1 `
  --stream-speed 86400 `
  --dry-run
```

**Output**: ~30 data points, 1 point setiap 1 detik

## Parameter Guide

### `--stream-interval` (detik real)

Interval pengiriman data dalam detik **realtime**.

- `1`: Sangat cepat, cocok untuk testing
- `3-5`: Moderate, cocok untuk demo
- `10+`: Lambat, cocok untuk long-running simulation

### `--stream-speed` (multiplier)

Berapa detik **simulasi** per 1 detik **real**.

**Formula**:

```
total_simulated_hours = 30 days × 24 hours = 720 hours
streaming_duration_seconds = (720 × 3600) / stream_speed
data_points = streaming_duration_seconds / stream_interval
```

**Common Values**:

| stream-speed | Meaning          | 30 hari jadi |
| ------------ | ---------------- | ------------ |
| 86400        | 1 hari/detik     | 30 detik     |
| 43200        | 12 jam/detik     | 1 menit      |
| 21600        | 6 jam/detik      | 2 menit      |
| 8640         | 2.4 jam/detik    | 5 menit      |
| 4320         | 1.2 jam/detik    | 10 menit     |
| 2592         | 43.2 menit/detik | 1 jam        |
| 1296         | 21.6 menit/detik | 2 jam        |

## Payload Format

Data yang dikirim memiliki struktur lengkap dengan **overall metrics** dan **component-level metrics**:

```json
{
  "timestamp": "2025-09-15T14:30:00",
  "shift": "B",
  "machine_id": "FLEXO_104",
  "overall": {
    "produced_units": 18500,
    "good_units": 17020,
    "reject_units": 1480,
    "runtime_minutes": 1250.5,
    "downtime_minutes": 189.5,
    "design_speed_cpm": 250.0,
    "actual_speed_cpm": 210.0,
    "oee": 0.6754,
    "availability": 0.8684,
    "performance": 0.84,
    "quality": 0.92
  },
  "components": {
    "PRE_FEEDER": {
      "tension_dev_pct": 2.15,
      "feed_stops_hour": 1,
      "uptime_ratio": 0.868
    },
    "FEEDER": {
      "double_sheet_hour": 0,
      "vacuum_dev_pct": 3.42,
      "uptime_ratio": 0.868
    },
    "PRINTING": {
      "registration_error_mm": 0.145,
      "registration_error_max_mm": 0.218,
      "ink_viscosity_dev_pct": 2.85,
      "reject_rate_pct": 1.62,
      "performance_ratio": 0.84
    },
    "SLOTTER": {
      "miscut_pct": 0.95,
      "burr_mm": 0.062,
      "blade_life_used_pct": 45.8,
      "uptime_ratio": 0.851
    },
    "DOWN_STACKER": {
      "jam_hour": 0,
      "misstack_pct": 0.48,
      "sync_dev_pct": 1.85,
      "uptime_ratio": 0.885
    }
  }
}
```

## Component Metrics Generation

Metrik komponen di-generate berdasarkan **OEE trend** dari data historis:

### Strategy:

- **Low Availability** → Higher `tension_dev_pct`, `feed_stops_hour` (PRE_FEEDER/FEEDER issues)
- **Low Performance** → Higher `registration_error_mm`, `ink_viscosity_dev_pct` (PRINTING speed issues)
- **Low Quality** → Higher `reject_rate_pct`, `miscut_pct` (PRINTING/SLOTTER defects)

### Realistic Noise:

Setiap metrik ditambahkan Gaussian noise ±5-30% untuk variasi natural.

## Troubleshooting

### Issue 1: "Kolom FLEXO 104 tidak ditemukan"

**Solution**: File Excel harus memiliki sheet produksi bulanan dengan kolom "FLEXO 4" atau "FLEXO 4 2 SHIFT"

**Note**: FLEXO 104 = FLEXO 4 (naming convention pabrik)

### Issue 2: "Tidak ada data untuk di-stream"

**Possible causes**:

- Sheet bulan tidak terdeteksi (harus ada SEPTEMBER, OKTOBER, dll di nama sheet)
- Kolom tanggal kosong
- Nilai produksi semua 0

**Solution**: Run dengan `--verbose` untuk debug detail

### Issue 3: API connection error

**Solution**:

1. Pastikan API backend running di `http://localhost:8100`
2. Test dengan `--dry-run` dulu
3. Check firewall/network

## Advanced Usage

### Custom Speed Calculation

Jika ingin durasi streaming spesifik:

```python
# Misal: ingin 1 bulan dalam 15 menit
desired_duration_minutes = 15
total_simulated_seconds = 30 * 24 * 3600  # 30 hari
desired_duration_seconds = desired_duration_minutes * 60

stream_speed = total_simulated_seconds / desired_duration_seconds
# = 2592000 / 900 = 2880
```

Then run:

```powershell
python sensor_simulator.py --file "..." --stream --stream-speed 2880 --stream-interval 3
```

### Multiple Files Sequential

```powershell
# Stream SEPTEMBER lalu OKTOBER
python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream --stream-speed 4320 --stream-interval 5

python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 10 OKTOBER 2025.xlsx" `
  --stream --stream-speed 4320 --stream-interval 5
```

## Best Practices

1. **Start with dry-run**: Selalu test dengan `--dry-run` dulu
2. **Use verbose for debugging**: Tambahkan `--verbose` jika ada issue
3. **Match API capacity**: Jangan set interval terlalu cepat jika API lambat
4. **Monitor API logs**: Check API menerima data dengan benar
5. **Test end-to-end**: Pastikan data muncul di dashboard

## Integration Example

### Terminal 1: Start API

```powershell
cd Model/05_API
uvicorn flexotwin_api:app --host 0.0.0.0 --port 8100 --reload
```

### Terminal 2: Run Simulator

```powershell
cd Model/06_SensorSimulator
python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream `
  --stream-interval 5 `
  --stream-speed 4320
```

### Terminal 3: Monitor Dashboard

```powershell
# Open browser
start http://localhost:5173
```

**Result**: Dashboard akan menampilkan data OEE + component health yang update setiap 5 detik! 🎉

## FAQ

**Q: Kenapa perlu streaming mode? Kenapa tidak kirim semua data sekaligus?**

A: Untuk simulasi kondisi **realtime**. Dashboard/API dirancang untuk handle data stream berkelanjutan, bukan batch upload.

**Q: Apakah bisa pause/resume streaming?**

A: Belum support native. Workaround: Ctrl+C untuk stop, lalu run lagi dengan file yang sama.

**Q: Apakah component metrics akurat?**

A: Metrik di-generate berdasarkan **rule-based estimation** dari OEE trend. Untuk data sensor real, perlu integrasi sensor fisik.

**Q: Bisa stream multiple machines sekaligus?**

A: Belum support. Run multiple instances dengan `--machine` berbeda (need implementation).

---

**Happy Streaming!** 🚀
