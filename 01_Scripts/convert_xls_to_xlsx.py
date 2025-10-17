"""
Batch Converter: .xls → .xlsx (Windows-friendly)

Fungsi:
- Memindai folder sumber dan mengonversi semua file .xls menjadi .xlsx ke folder tujuan baru.
- Prioritas metode: Microsoft Excel COM (win32com) → akurat dan stabil untuk .xls lama.
- Jika COM tidak tersedia, tampilkan instruksi alternatif (pyexcel) agar pengguna bisa melanjutkan.

Catatan:
- xlrd>=2.0 tidak mendukung .xls, jadi pandas+xlsrd modern tidak bisa membaca .xls.
- Jalankan di Windows dengan Excel terinstal untuk hasil terbaik.

Penggunaan:
  python convert_xls_to_xlsx.py \
    --input-dir "..\\08_Data Produksi" \
    --output-dir "..\\08_Data Produksi\\converted_xlsx" \
    [--copy-xlsx]

Contoh cepat (default path otomatis ke 08_Data Produksi):
  python convert_xls_to_xlsx.py
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Optional


def default_paths():
    # Script berada di: Model/01_Scripts/convert_xls_to_xlsx.py
    # Default input:    Model/08_Data Produksi
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, ".."))
    input_dir = os.path.join(model_dir, "08_Data Produksi")
    output_dir = os.path.join(input_dir, "converted_xlsx")
    return input_dir, output_dir


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_xls(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".xls") and not path.lower().endswith(".xlsx")


def is_xlsx(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".xlsx")


def convert_with_com(src: str, dst: str) -> bool:
    """Convert using Excel COM automation. Returns True on success."""
    try:
        import win32com.client as win32  # type: ignore
    except Exception as e:
        print("[!] win32com tidak tersedia. Install dengan: pip install pywin32")
        return False

    excel = None
    try:
        excel = win32.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        print(f"[COM] Membuka: {src}")
        wb = excel.Workbooks.Open(os.path.abspath(src))
        # FileFormat=51 => xlOpenXMLWorkbook (.xlsx)
        wb.SaveAs(os.path.abspath(dst), FileFormat=51)
        wb.Close(SaveChanges=False)
        print(f"[COM] Tersimpan: {dst}")
        return True
    except Exception as e:
        print(f"[COM] Gagal konversi {src}: {e}")
        return False
    finally:
        try:
            if excel is not None:
                excel.Quit()
                time.sleep(0.5)
        except Exception:
            pass


def copy_xlsx(src: str, dst: str) -> bool:
    try:
        import shutil
        shutil.copy2(src, dst)
        print(f"[COPY] {src} → {dst}")
        return True
    except Exception as e:
        print(f"[COPY] Gagal menyalin {src}: {e}")
        return False


def main():
    d_input, d_output = default_paths()

    ap = argparse.ArgumentParser(description="Batch convert .xls to .xlsx")
    ap.add_argument("--input-dir", default=d_input, help="Folder sumber .xls")
    ap.add_argument("--output-dir", default=d_output, help="Folder tujuan .xlsx")
    ap.add_argument("--copy-xlsx", action="store_true", help="Ikut salin file .xlsx yang sudah ada ke folder tujuan")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    ensure_dir(out_dir)

    if not os.path.isdir(in_dir):
        print(f"[X] Folder input tidak ditemukan: {in_dir}")
        sys.exit(1)

    # List files
    entries = sorted(os.listdir(in_dir))
    xls_files = [os.path.join(in_dir, f) for f in entries if is_xls(os.path.join(in_dir, f))]
    xlsx_files = [os.path.join(in_dir, f) for f in entries if is_xlsx(os.path.join(in_dir, f))]

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Ditemukan {len(xls_files)} file .xls untuk dikonversi")
    if args.copy_xlsx:
        print(f"Opsi copy .xlsx aktif: {len(xlsx_files)} file akan disalin juga")

    # Konversi XLS → XLSX via COM
    ok = 0
    fail = 0
    for src in xls_files:
        base = os.path.splitext(os.path.basename(src))[0] + ".xlsx"
        dst = os.path.join(out_dir, base)
        if convert_with_com(src, dst):
            ok += 1
        else:
            fail += 1

    # Opsional: salin .xlsx existing
    if args.copy_xlsx:
        for src in xlsx_files:
            base = os.path.basename(src)
            dst = os.path.join(out_dir, base)
            copy_xlsx(src, dst)

    print("\nRingkasan:")
    print(f"  Berhasil : {ok}")
    print(f"  Gagal    : {fail}")
    if fail > 0:
        print("\nSaran bila gagal:")
        print("- Pastikan Microsoft Excel terinstal.")
        print("- Pastikan package pywin32 terinstal: pip install pywin32")
        print("- Jika tidak ada Excel, gunakan alternatif: pyexcel + pyexcel-xls + pyexcel-xlsx")
        print("  pip install pyexcel pyexcel-xls pyexcel-xlsx")
        print("  (Kemudian implementasi fallback dapat ditambahkan jika diperlukan)")


if __name__ == "__main__":
    main()
