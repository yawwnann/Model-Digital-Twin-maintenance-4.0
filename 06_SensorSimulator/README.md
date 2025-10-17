# FLEXO Sensor Simulator

Simulator sensor untuk Digital Twin mesin FLEXO yang membaca data historis dari file Excel dan mengirimkannya secara realtime ke API backend.

## 📋 Fitur

- ✅ Membaca data produksi, OEE, dan losstime dari file XLSX
- ✅ Filter otomatis untuk mesin FLEXO 104
- ✅ Ekstraksi multi-sheet (OEE, Produksi Bulanan, Losstime)
- ✅ Deteksi kolom dinamis
- ✅ HTTP POST ke Realtime OEE API
- ✅ Dry-run mode untuk testing
- ✅ Configurable interval streaming
- ✅ Mode continuous loop
- ✅ Verbose logging untuk debugging

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas openpyxl requests
```

### Basic Usage

#### 1. Dry Run (Testing tanpa kirim data)

```powershell
python sensor_simulator.py --dry-run
```

#### 2. Kirim data dari satu file

```powershell
python sensor_simulator.py --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx"
```

#### 3. Proses semua file di folder

```powershell
python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx"
```

#### 4. Mode continuous dengan interval custom

```powershell
python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx" --loop --interval 5
```

## 📖 Command Line Options

| Option         | Type   | Default                                    | Description                       |
| -------------- | ------ | ------------------------------------------ | --------------------------------- |
| `--file`       | string | -                                          | Path ke file Excel tunggal        |
| `--folder`     | string | `../08_Data Produksi/data_xlsx`            | Path ke folder berisi file Excel  |
| `--machine`    | string | `FLEXO_104`                                | Machine ID untuk identifikasi     |
| `--api`        | string | `http://localhost:8100/ingest/sensor-data` | API endpoint target               |
| `--interval`   | float  | `1.0`                                      | Delay antar pengiriman (detik)    |
| `--max-points` | int    | `None`                                     | Maksimal data points per file     |
| `--max-files`  | int    | `None`                                     | Maksimal file yang diproses       |
| `--loop`       | flag   | `False`                                    | Loop continuous setelah selesai   |
| `--dry-run`    | flag   | `False`                                    | Testing mode (tidak kirim ke API) |
| `--verbose`    | flag   | `False`                                    | Verbose logging untuk debugging   |

## 📊 Data Extraction

Simulator mengekstrak data dari beberapa sheet:

### 1. Sheet OEE

- Design Speed
- OEE (%)
- Availability (%)
- Performance (%)
- Quality (%)

### 2. Sheet Produksi Bulanan (SEPTEMBER, OKTOBER, dll)

- Tanggal produksi
- Produced units
- Actual speed
- Shift information

### 3. Sheet LOSSTIME

- Downtime per tanggal
- Losstime reasons
- Total menit downtime

## 🔧 API Payload Format

Simulator mengirim data dalam format JSON:

```json
{
  "machine_id": "FLEXO_104",
  "timestamp": "2025-09-01T08:00:00",
  "produced_units": 1000,
  "good_units": 950,
  "reject_units": 50,
  "runtime_minutes": 480,
  "downtime_minutes": 60,
  "design_speed": 200,
  "actual_speed": 180,
  "shift": 1
}
```

API akan menghitung OEE metrics (Availability, Performance, Quality, OEE) secara otomatis.

## 📝 Examples

### Example 1: Development Testing

```powershell
# Test parsing tanpa kirim data
python sensor_simulator.py --file "test_data.xlsx" --dry-run --verbose
```

### Example 2: Single File Production

```powershell
# Kirim data dari file September dengan interval 2 detik
python sensor_simulator.py --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" --interval 2
```

### Example 3: Batch Processing

```powershell
# Proses 5 file pertama saja
python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx" --max-files 5
```

### Example 4: Continuous Streaming

```powershell
# Loop terus menerus dengan interval 10 detik
python sensor_simulator.py --folder "../08_Data Produksi/data_xlsx" --loop --interval 10
```

### Example 5: Custom API Endpoint

```powershell
# Kirim ke API production
python sensor_simulator.py --api "http://production-server:8080/api/sensor-data" --folder "../08_Data Produksi/data_xlsx"
```

## 🐛 Troubleshooting

### Issue: "Folder tidak ditemukan"

**Solution**: Pastikan path folder benar. Gunakan path relative dari lokasi script atau absolute path:

```powershell
python sensor_simulator.py --folder "C:\Users\HP\Documents\Belajar python\Digital Twin\Web\Model\08_Data Produksi\data_xlsx"
```

### Issue: "Tidak ada data yang bisa diekstrak"

**Solution**: Jalankan dengan `--verbose` untuk melihat detail parsing:

```powershell
python sensor_simulator.py --file "data.xlsx" --dry-run --verbose
```

### Issue: "API returned status 500"

**Solution**: Pastikan Realtime OEE API sudah running:

```powershell
cd ../05_API
uvicorn realtime_oee_api:app --host 0.0.0.0 --port 8100 --reload
```

### Issue: "Failed to send data: Connection refused"

**Solution**:

1. Cek API endpoint dengan `--verbose`
2. Pastikan port 8100 tidak diblok firewall
3. Test dengan dry-run dulu: `--dry-run`

## 📚 Advanced Usage

### Custom Machine ID

Jika ingin simulate mesin lain:

```powershell
python sensor_simulator.py --machine "FLEXO_102" --folder "../08_Data Produksi/data_xlsx"
```

### Rate Limiting

Untuk menghindari overload API:

```powershell
python sensor_simulator.py --interval 5 --max-points 10
```

### Debugging

Untuk melihat detail ekstraksi data:

```powershell
python sensor_simulator.py --dry-run --verbose --file "test.xlsx"
```

## 🔄 Integration with Backend

Simulator ini dirancang untuk bekerja dengan **Realtime OEE API**:

```
Sensor Simulator → POST /ingest/sensor-data → SQLite → Dashboard
```

Pastikan API sudah running sebelum menjalankan simulator (lihat `../05_API/README.md`).

## 📦 Dependencies

- `pandas >= 2.1.3` - Data manipulation
- `openpyxl >= 3.1.2` - Excel file reading
- `requests >= 2.31.0` - HTTP client

Install semua:

```bash
pip install -r requirements.txt
```

## 🎯 Next Steps

1. ✅ Test dengan dry-run
2. ✅ Validasi data extraction
3. ✅ Jalankan API backend
4. ✅ Run simulator → API
5. ✅ Monitor dashboard realtime

## 📞 Support

Jika ada masalah atau pertanyaan, jalankan dengan `--verbose` dan periksa log output untuk debugging.
