# 🚀 Quick Start Guide - Digital Twin Dashboard

## Streamlit Dashboard (Tanpa Database)

Dashboard realtime untuk monitoring OEE, component health, dan digital twin visualization.

---

## Prerequisites

```powershell
# Install dependencies
pip install streamlit plotly pandas
pip install openpyxl requests  # Untuk simulator
```

---

## Cara Menjalankan

### Step 1: Jalankan Dashboard

```powershell
cd "c:\Users\HP\Documents\Belajar python\Digital Twin\Web\Model\06_SensorSimulator"

streamlit run streamlit_dashboard.py --server.port 8501
```

**Output:**

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 2: Buka Browser

```
http://localhost:8501
```

Dashboard akan tampil dengan status "⏳ Waiting for sensor data..."

### Step 3: Jalankan Simulator (Terminal Baru)

```powershell
cd "c:\Users\HP\Documents\Belajar python\Digital Twin\Web\Model\06_SensorSimulator"

python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream `
  --stream-interval 5 `
  --sim-interval 1.0 `
  --api "file://latest_data.json"
```

**Penjelasan Parameter:**

- `--stream`: Aktifkan streaming mode
- `--stream-interval 5`: Kirim data setiap 5 detik
- `--sim-interval 1.0`: 1 data point per jam simulasi (720 points untuk 30 hari)
- `--api "file://latest_data.json"`: Simpan ke file (untuk Streamlit)

### Step 4: Monitor Dashboard

Dashboard akan **auto-refresh setiap 5 detik** dan menampilkan:

✅ **Key Metrics**: Overall Health, OEE, Availability, Performance, Quality  
✅ **OEE Gauges**: Visual gauge untuk A, P, Q  
✅ **Digital Twin**: Flow diagram komponen dengan color-coded health  
✅ **Component Health**: Bar chart dan detail per komponen  
✅ **Historical Trends**: Line chart OEE over time  
✅ **Recommendations**: Alert dan saran maintenance

---

## Parameter Tuning

### Interval Streaming

| Parameter              | Efek                   | Use Case                   |
| ---------------------- | ---------------------- | -------------------------- |
| `--stream-interval 1`  | Update setiap 1 detik  | Testing cepat              |
| `--stream-interval 5`  | Update setiap 5 detik  | **Recommended untuk demo** |
| `--stream-interval 10` | Update setiap 10 detik | Long-running simulation    |

### Simulation Interval

| Parameter            | Data Points             | Durasi Total (5s interval) | Use Case                              |
| -------------------- | ----------------------- | -------------------------- | ------------------------------------- |
| `--sim-interval 1.0` | 720 (1 jam per point)   | 60 menit                   | **Recommended (1 bulan dalam 1 jam)** |
| `--sim-interval 0.5` | 1440 (30 min per point) | 120 menit                  | Detail analysis                       |
| `--sim-interval 2.0` | 360 (2 jam per point)   | 30 menit                   | Quick demo                            |

---

## Troubleshooting

### Issue 1: "Waiting for sensor data..."

**Cause:** Simulator belum jalan atau file `latest_data.json` belum dibuat

**Solution:**

1. Check simulator running di terminal lain
2. Check file `latest_data.json` ada di folder simulator
3. Pastikan parameter `--api "file://latest_data.json"` benar

### Issue 2: Dashboard tidak update

**Cause:** Auto-refresh disabled

**Solution:** Centang checkbox "Auto Refresh (5s)" di kanan atas

### Issue 3: "component_health_calculator.py tidak ditemukan"

**Cause:** File calculator tidak ada atau path salah

**Solution:**

```powershell
# Check file ada
ls component_health_calculator.py

# Pastikan di folder yang sama dengan streamlit_dashboard.py
```

### Issue 4: Simulator error "Kolom FLEXO 104 tidak ditemukan"

**Cause:** File Excel tidak ada atau struktur berbeda

**Solution:** Gunakan file yang sudah tested:

```powershell
python sensor_simulator.py `
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" `
  --stream --stream-interval 5 --sim-interval 1.0 --api "file://latest_data.json"
```

---

## Architecture

```
┌─────────────────────┐
│  Excel Data (XLSX)  │
│  (30 hari produksi) │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────────────────────────────────────┐
│  sensor_simulator.py                                │
│  - Extract daily data (20-30 points)                │
│  - Interpolate to hourly (720 points)               │
│  - Generate component metrics                       │
│  - Stream every 5s → latest_data.json               │
└──────────┬──────────────────────────────────────────┘
           │ (file write)
           ↓
┌──────────────────────────────┐
│  latest_data.json            │
│  {timestamp, overall,        │
│   components}                │
└──────────┬───────────────────┘
           │ (file read, every 5s)
           ↓
┌─────────────────────────────────────────────────────┐
│  streamlit_dashboard.py                             │
│  - Read latest_data.json                            │
│  - Calculate health (component_health_calculator)   │
│  - Visualize: OEE gauges, component health,         │
│    digital twin, trends                             │
│  - Auto-refresh every 5s                            │
└─────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────┐
│  Browser (localhost:8501)    │
│  Interactive Dashboard       │
└──────────────────────────────┘
```

---

## Features Checklist

✅ Real-time OEE monitoring  
✅ Component health scoring (rule-based)  
✅ Digital twin visualization  
✅ Historical trends (in-memory, max 1000 points)  
✅ Auto-refresh dashboard  
✅ Alert & recommendations  
✅ Production summary  
✅ No database required  
✅ Easy setup & deploy

---

## Advanced Usage

### Multiple Files Sequential

Terminal 1: Dashboard

```powershell
streamlit run streamlit_dashboard.py
```

Terminal 2: Stream September

```powershell
python sensor_simulator.py --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" --stream --stream-interval 5 --sim-interval 1.0 --api "file://latest_data.json"
```

**Setelah selesai**, stream Oktober:

```powershell
python sensor_simulator.py --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 10 OKTOBER 2025.xlsx" --stream --stream-interval 5 --sim-interval 1.0 --api "file://latest_data.json"
```

### Custom Dashboard Theme

Edit `streamlit_dashboard.py` bagian CSS (line ~75) untuk customize colors.

### Export Data

Dashboard menyimpan buffer (max 1000 points) in-memory. Untuk export:

1. Tambahkan button di sidebar:

```python
if st.button("Export to CSV"):
    df = pd.DataFrame(list(st.session_state.data_buffer))
    df.to_csv('export_data.csv', index=False)
    st.success("Data exported!")
```

---

## Next Steps

### Untuk Production

Jika butuh:

- ✅ Persistent storage → Implement SQLite/PostgreSQL
- ✅ Multi-user access → Deploy dengan authentication
- ✅ Historical analysis → Add database dengan time-series optimization
- ✅ Alert notification → Add email/Slack integration
- ✅ API endpoints → Migrate ke FastAPI + WebSocket

### Untuk Development

- [ ] Add unit tests
- [ ] Add component detail modal
- [ ] Add export functionality
- [ ] Add configuration panel
- [ ] Add machine selection (multi-machine)

---

## Support

**Issues?** Check:

1. Python version >= 3.8
2. All dependencies installed
3. File paths correct (relative/absolute)
4. Excel file has FLEXO 4 column
5. Firewall not blocking localhost

**Documentation:**

- STREAMING_GUIDE.md - Detail simulator streaming mode
- COMPONENT_HEALTH_SPECIFICATION.md - Health assessment rules
- README.md - Simulator usage

---

**Happy Monitoring!** 🎉
