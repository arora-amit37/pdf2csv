# ===================== IMPORTS =====================
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import os
import re
import pandas as pd
import glob
import pickle
import multiprocessing as mp

# Optional imports you may need for images (not used in core logic, but kept if needed later)
from PIL import Image, ImageFilter, ImageOps
import io
import csv
from thefuzz import process  # pip install thefuzz

# ===================== PATHS =====================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_bin_path = r'C:\poppler\poppler-25.12.0\Library\bin'

PDF_DIR = r'C:\pdf2csv\extracted_files\recovered'
CHECKPOINT_FILE = r'C:\pdf2csv\checkpoint.pkl'
CHECKPOINT_EVERY = 100
WORKERS = max(mp.cpu_count() - 1, 1)

# ===================== COLUMNS =====================
target_columns = [
    'source_pdf', 'row_in_pdf',
    'lender', 'clsing_dt', 'endrsmt_dt', 'tmntn_dt', 'Borr_Age',
    'Coborr_Age', 'Borr_Cnt', 'es_status', 'int_rt', 'int_rt_10yr',
    'hecm_margin', 'pd_stmln_flg', 'rt_typ', 'arm_indx_typ',
    'arm_prdc_typ', 'max_clm_amt', 'init_prncpl_lmt',
    'hecm_orgntn_fees', 'prop_addr_zip_cd', 'loan_typ'
]

# ===================== ROBUST PARSER =====================
def robust_parse(line):
    row = [None] * 20

    # ---------- OCR NORMALIZATION ----------
    line = (
        line.replace('”', '"')
            .replace("’", "'")
            .replace('!', '')
            .replace('|', ' ')
    )

    # ---------- PATTERN EXTRACTION ----------
    dates = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', line)
    decimals = re.findall(r'\d+\.\d+', line)

    clean_text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' ', line)
    clean_text = re.sub(r'\d+\.\d+', ' ', clean_text)
    numbers = re.findall(r'\b\d+\b', clean_text)

    # ---------- LENDER ----------
    lender_match = re.search(r'^(.*?)(?=\d{1,2}/)', line)
    row[0] = lender_match.group(1).strip().strip('"').strip("'") if lender_match else None

    # ---------- DATES ----------
    if len(dates) >= 1: row[1] = dates[0]
    if len(dates) >= 2: row[2] = dates[1]
    if len(dates) >= 3: row[3] = dates[2]

    # ---------- STATUS ----------
    if re.search(r'Termin[a-z]*', line, re.I):
        row[7] = "Terminated"
    elif re.search(r'Active', line, re.I):
        row[7] = "Active"

    # ---------- DECIMALS ----------
    if len(decimals) >= 1: row[8] = decimals[0]
    if len(decimals) >= 2: row[9] = decimals[1]
    if len(decimals) >= 3: row[10] = decimals[2]

    # ---------- AGES & COUNT ----------
    if row[3] and row[7]:
        try:
            start = line.index(row[3]) + len(row[3])
            end = line.lower().index(row[7].lower())
            nums = re.findall(r'\b\d+\b', line[start:end])
            if len(nums) == 3:
                row[4], row[5], row[6] = nums
            elif len(nums) == 2:
                row[4], row[6] = nums
        except ValueError:
            pass

    # ---------- FLAGS ----------
    flags = re.findall(r'\b[A-Z]\b', line.upper())
    if len(flags) >= 1: row[11] = flags[0]
    if len(flags) >= 2: row[12] = flags[1]
    if len(flags) >= 3: row[13] = flags[2]

    # ---------- ARM PRODUCT ----------
    arm_match = re.search(r'\b\d+[A-Z]\b', line)
    row[14] = arm_match.group(0) if arm_match else None

    # ---------- MAX CLAIM & PRINCIPAL ----------
    large_nums = [n for n in numbers if len(n) > 3]
    if len(large_nums) >= 1: row[15] = large_nums[0]
    if len(large_nums) >= 2: row[16] = large_nums[1]

    # ---------- FEES ----------
    row[17] = 0

    # ---------- ZIP & LOAN TYPE ----------
    if len(numbers) >= 2:
        row[19] = str(numbers[-1]).zfill(2)
        row[18] = numbers[-2] if len(numbers[-2]) == 5 else None

    return row

# ===================== TEXT EXTRACTION =====================
def extract_text(pdf_path):
    try:
        # Try pdfplumber first (fast path for digital PDFs)
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text()
            if text and len(text.strip()) > 50:
                return text
    except Exception:
        pass

    # If pdfplumber fails or blank → OCR
    images = convert_from_path(pdf_path, first_page=1, last_page=1, poppler_path=poppler_bin_path)
    return pytesseract.image_to_string(images[0], config='--oem 3 --psm 6')

# ===================== PROCESS SINGLE PDF =====================
def process_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        lines = [l for l in text.split('\n') if len(l.strip()) > 20]
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')

        rows = []
        for i, line in enumerate(lines, start=1):
            parsed = robust_parse(line)
            rows.append([pdf_name, i] + parsed)

        return rows

    except Exception as e:
        print(f"FAILED: {pdf_path} -> {e}")
        return []

# ===================== CHECKPOINTING =====================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return set(), []

def save_checkpoint(done_files, all_rows):
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump((done_files, all_rows), f)

# ===================== MAIN EXECUTION =====================
if __name__ == '__main__':
    pdf_files = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
    done_files, all_rows = load_checkpoint()
    remaining = [p for p in pdf_files if p not in done_files]

    print(f"Remaining PDFs: {len(remaining)}")
    pool = mp.Pool(WORKERS)

    for idx, result in enumerate(pool.imap_unordered(process_pdf, remaining), 1):
        pdf_path = remaining[idx - 1]
        done_files.add(pdf_path)
        all_rows.extend(result)

        if idx % CHECKPOINT_EVERY == 0:
            save_checkpoint(done_files, all_rows)
            print(f"Checkpoint saved at {idx} files")

    pool.close()
    pool.join()

    save_checkpoint(done_files, all_rows)
    master_df = pd.DataFrame(all_rows, columns=target_columns)

    # Save final CSV
    master_df.to_csv(r'C:\pdf2csv\master_output.csv', index=False)
    print("Processing complete! CSV saved as master_output.csv")