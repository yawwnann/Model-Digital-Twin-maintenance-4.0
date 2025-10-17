"""
Profiler untuk file XLSX hasil konversi.

Tujuan:
- Memindai folder converted_xlsx
- Menemukan sheet yang relevan dengan FLEXO 104 (berdasarkan teks di sheet name atau isi kolom)
- Mendaftar kolom yang tersedia dan sample 3 baris pertama
- Membantu menentukan pemetaan kolom untuk sensor simulator

Cara pakai:
  python profile_converted_xlsx.py --folder "..\\08_Data Produksi\\converted_xlsx"
"""

import argparse
import os
import pandas as pd


def find_folder_default():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "08_Data Produksi", "data_xlsx"))


def load_excel_sheets(path):
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        return xl, xl.sheet_names
    except Exception as e:
        print(f"[X] Gagal membaca {path}: {e}")
        return None, []


def sheet_relevant_to_flexo104(name: str) -> bool:
    n = (name or "").lower()
    # Nama sheet yang sering muncul: OEE, produksi, tinta/ink, losstime, achievement, dll.
    # Kita tidak batasi terlalu ketat, karena filter FLEXO 104 dilakukan pada isi.
    return any(k in n for k in ["oee", "produksi", "production", "ink", "tinta", "losstime", "downtime", "achievement", "data", "sheet"]) or True


def df_contains_flexo104(df: pd.DataFrame) -> bool:
    # Cek apakah ada kolom yang mengandung kata FLEXO; atau ada nilai yang mengandung 'FLEXO 104'
    cols = [c for c in df.columns if isinstance(c, str)]
    has_flexo_col = any("flexo" in c.lower() for c in cols)
    if has_flexo_col:
        return True
    # Cek beberapa kolom teks umum
    for c in cols[:10]:
        try:
            ser = df[c].astype(str).str.upper()
            if ser.str.contains("FLEXO 104", na=False).any():
                return True
        except Exception:
            continue
    return False


def summarize_df(df: pd.DataFrame, max_rows: int = 3):
    # Ambil 3 baris pertama yang tidak kosong
    preview = df.head(max_rows)
    return preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=find_folder_default(), help="Folder converted_xlsx")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"[X] Folder tidak ditemukan: {folder}")
        return

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".xlsx")]
    files.sort()
    if not files:
        print("[!] Tidak ditemukan file .xlsx di folder ini.")
        return

    print(f"Memindai {len(files)} file di {folder}\n")

    for path in files:
        print(f"==== {os.path.basename(path)} ====")
        xl, sheets = load_excel_sheets(path)
        if not xl:
            continue
        print("Sheets:", ", ".join(sheets))
        for sn in sheets:
            try:
                if not sheet_relevant_to_flexo104(sn):
                    continue
                df = xl.parse(sn)
                if df.empty:
                    continue
                # Hanya display jika mengandung FLEXO 104
                if df_contains_flexo104(df) or True:
                    cols = list(df.columns)
                    print(f"  - Sheet: {sn}")
                    print(f"    Kolom ({len(cols)}): {cols[:12]}{'...' if len(cols)>12 else ''}")
                    prev = summarize_df(df)
                    with pd.option_context('display.max_columns', None):
                        print(prev.to_string(index=False))
            except Exception as e:
                print(f"    [!] Gagal baca sheet {sn}: {e}")
        print()


if __name__ == "__main__":
    main()
