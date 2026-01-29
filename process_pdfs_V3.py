# ===================== IMPORTS =====================
import pytesseract
from pdf2image import convert_from_path
import os
import re
import pandas as pd
import glob
import pickle
import multiprocessing as mp
from datetime import datetime

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import csv

# ===================== PATHS =====================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_bin_path = r'C:\poppler\poppler-25.12.0\Library\bin'

PDF_DIR = r'C:\pdf2csv\extracted_files\recovered'

CHECKPOINT_FILE = r'C:\pdf2csv\checkpoint_v3_optimized.pkl'

CHECKPOINT_EVERY = 50  # Checkpoint every 50 files as requested
WORKERS = max(mp.cpu_count() - 1, 1)

# ===================== COLUMNS =====================
target_columns = [
    'source_pdf', 'row_in_pdf', 'raw_text',
    'lender', 'clsng_dt', 'endrsmt_dt', 'tmntn_dt', 'Borr_Age',
    'Coborr_Age', 'Borr_Cnt', 'es_status', 'int_rt', 'int_rt_10yr',
    'hecm_margin', 'pd_stmln_flg', 'rt_typ', 'arm_indx_typ',
    'arm_prdc_typ', 'max_clm_amt', 'init_prncpl_lmt',
    'hecm_orgntn_fees', 'prop_addr_zip_cd', 'loan_typ'
]

# ===================== VALIDATION RULES =====================
VALID_VALUES = {
    'pd_stmln_flg': ['N', 'C', 'R', 'H'],
    'rt_typ': ['A', 'M', 'F'],
    'arm_indx_typ': ['T', 'L', ''],
    'arm_prdc_typ': ['1Y', '1M', ''],
    'loan_typ': ['02', '01', '04', '05', '03', '06'],
    'es_status': ['Terminated', 'Assigned', 'Endorsed', '']
}

# ===================== IMAGE PREPROCESSING =====================
def preprocess_image(image):
    """Enhanced preprocessing for better OCR accuracy"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    
    # Denoise with median filter
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # Binarization (convert to black and white)
    threshold = 128
    image = image.point(lambda p: p > threshold and 255)
    
    # Scale up for better OCR (2x)
    width, height = image.size
    image = image.resize((width * 2, height * 2), Image.LANCZOS)
    
    return image

# ===================== VALIDATION HELPERS =====================
def validate_date(date_str):
    """Validate and standardize date format M/D/YYYY"""
    if not date_str:
        return None
    
    # Match M(1-2)/D(1-2)/YYYY(4)
    match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str.strip())
    if match:
        month, day, year = match.groups()
        # Basic validation
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{month}/{day}/{year}"
    return None

def validate_decimal(value, min_val, max_val, max_decimals):
    """Validate decimal is within range and has proper format"""
    if not value:
        return None
    try:
        val = float(value)
        if min_val <= val <= max_val:
            # Check decimal places
            decimal_part = str(value).split('.')[-1] if '.' in str(value) else ''
            if len(decimal_part) <= max_decimals:
                return value
    except:
        pass
    return None

def validate_age(age_str):
    """Validate age is numeric and mostly >60"""
    if not age_str:
        return None
    try:
        age = int(age_str)
        if 18 <= age <= 120:  # Reasonable age range
            return str(age)
    except:
        pass
    return None

def validate_zip(zip_str):
    """Validate 5-digit zip code"""
    if not zip_str:
        return None
    cleaned = re.sub(r'[^\d]', '', zip_str)
    if len(cleaned) == 5:
        return cleaned
    return None

# ===================== ROBUST PARSER =====================
def robust_parse(line):
    """Enhanced parsing with strict validation"""
    row = [None] * 20
    
    # Clean common OCR errors
    line = (
        line.replace('"', '"')
            .replace('"', '"')
            .replace("'", "'")
            .replace("'", "'")
            .replace('!', 'I')
            .replace('|', 'I')
            .replace('О', '0')  # Cyrillic O to zero
            .replace('о', '0')
            .replace('З', '3')
    )
    
    # Extract all dates (M/D/YYYY format)
    date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
    dates = re.findall(date_pattern, line)
    validated_dates = [validate_date(d) for d in dates]
    validated_dates = [d for d in validated_dates if d]
    
    # Extract all decimals
    decimal_pattern = r'\b\d+\.\d+\b'
    decimals = re.findall(decimal_pattern, line)
    
    # Remove dates and decimals for cleaner number extraction
    clean_text = re.sub(date_pattern, ' ', line)
    clean_text = re.sub(decimal_pattern, ' ', clean_text)
    
    # Extract integers
    numbers = re.findall(r'\b\d+\b', clean_text)
    
    # 1. LENDER - Large ALL CAPS text before first date, may have quotes/commas
    lender_match = re.search(r'^[\s\'"]*(.*?)(?=\d{1,2}/)', line)
    if lender_match:
        lender = lender_match.group(1).strip().strip('"\'').strip()
        # Ensure it's mostly uppercase or contains common lender patterns
        if lender and (lender.isupper() or len(lender) > 5):
            row[0] = lender
    
    # 2-4. DATES: clsng_dt, endrsmt_dt, tmntn_dt
    if len(validated_dates) >= 1:
        row[1] = validated_dates[0]
    if len(validated_dates) >= 2:
        row[2] = validated_dates[1]
    if len(validated_dates) >= 3:
        row[3] = validated_dates[2]
    
    # 8. ES_STATUS - Must be one of allowed values
    status_lower = line.lower()
    if 'terminated' in status_lower or 'termin' in status_lower:
        row[7] = "Terminated"
    elif 'assigned' in status_lower:
        row[7] = "Assigned"
    elif 'endorsed' in status_lower:
        row[7] = "Endorsed"
    else:
        row[7] = ""
    
    # 9-11. DECIMAL RATES
    # int_rt: 1-10, up to 3 decimals
    if len(decimals) >= 1:
        row[8] = validate_decimal(decimals[0], 1, 10, 3)
    # int_rt_10yr: 1-10, up to 3 decimals
    if len(decimals) >= 2:
        row[9] = validate_decimal(decimals[1], 1, 10, 3)
    # hecm_margin: 1-5, up to 2 decimals
    if len(decimals) >= 3:
        row[10] = validate_decimal(decimals[2], 1, 5, 2)
    
    # 5-7. AGES and BORROWER COUNT
    # Extract ages between dates[2] (tmntn_dt) and status
    if row[3] or row[7]:
        try:
            # Find section between third date and status
            start_idx = 0
            if row[3]:
                start_idx = line.index(validated_dates[2]) + len(validated_dates[2]) if len(validated_dates) >= 3 else 0
            
            end_idx = len(line)
            if row[7]:
                end_idx = line.lower().index(row[7].lower())
            
            age_section = line[start_idx:end_idx]
            age_nums = re.findall(r'\b\d+\b', age_section)
            
            # Filter for reasonable ages
            valid_ages = [n for n in age_nums if validate_age(n)]
            
            if len(valid_ages) >= 1:
                row[4] = validate_age(valid_ages[0])  # Borr_Age
            if len(valid_ages) >= 2:
                row[5] = validate_age(valid_ages[1])  # Coborr_Age
            
            # Borr_Cnt: 1 if only Borr_Age, 2 if both ages present
            if row[4] and row[5]:
                row[6] = '2'
            elif row[4]:
                row[6] = '1'
        except:
            pass
    
    # 12-14. SINGLE LETTER FLAGS
    # Extract single uppercase letters
    single_letters = re.findall(r'\b([A-Z])\b', line)
    
    # pd_stmln_flg: N, C, R, H (usually N)
    for letter in single_letters:
        if letter in VALID_VALUES['pd_stmln_flg']:
            row[11] = letter
            break
    
    # rt_typ: A, M, F
    for letter in single_letters:
        if letter in VALID_VALUES['rt_typ'] and letter != row[11]:
            row[12] = letter
            break
    
    # arm_indx_typ: T, L, or blank
    for letter in single_letters:
        if letter in VALID_VALUES['arm_indx_typ'] and letter not in [row[11], row[12]]:
            row[13] = letter
            break
    if row[13] is None:
        row[13] = ''
    
    # 15. ARM_PRDC_TYP: 1Y, 1M, or blank
    arm_match = re.search(r'\b(1[YM])\b', line)
    if arm_match:
        row[14] = arm_match.group(1)
    else:
        row[14] = ''
    
    # 16-18. LARGE NUMBERS
    large_nums = [n for n in numbers if len(n) >= 4]
    
    # max_clm_amt: numeric, often multiples of 1000
    if len(large_nums) >= 1:
        row[15] = large_nums[0]
    
    # init_prncpl_lmt: decimal
    if len(decimals) >= 4:
        row[16] = decimals[3]
    elif len(large_nums) >= 2:
        row[16] = large_nums[1]
    
    # hecm_orgntn_fees: numeric
    row[17] = '0'  # Default to 0 as in original
    if len(large_nums) >= 3:
        row[17] = large_nums[2]
    
    # 19-20. ZIP and LOAN TYPE (at end of line)
    # Extract last 5-digit number as zip
    zip_match = re.search(r"['\"]?(\d{5})['\"]?\s*['\"]?(\d{2})['\"]?\s*$", line)
    if zip_match:
        row[18] = validate_zip(zip_match.group(1))
        loan_type = zip_match.group(2)
        if loan_type in VALID_VALUES['loan_typ']:
            row[19] = loan_type
    else:
        # Try to find them separately
        if len(numbers) >= 2:
            # loan_typ: last 2-digit number
            if numbers[-1] in VALID_VALUES['loan_typ']:
                row[19] = numbers[-1]
            elif len(numbers[-1]) == 2 and numbers[-1] in VALID_VALUES['loan_typ']:
                row[19] = numbers[-1]
            
            # zip: second to last if 5 digits
            if len(numbers[-2]) == 5:
                row[18] = validate_zip(numbers[-2])
    
    return row

# ===================== OCR TEXT EXTRACTION =====================
def extract_text(pdf_path):
    """Extract text with enhanced preprocessing"""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=1,
            last_page=1,
            poppler_path=poppler_bin_path,
            dpi=300  # Higher DPI for better quality
        )
        
        # Preprocess image
        processed_img = preprocess_image(images[0])
        
        # OCR with optimized config
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/\'" ()'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        return text
    except Exception as e:
        print(f"OCR Error for {pdf_path}: {e}")
        return ""

# ===================== PROCESS SINGLE PDF =====================
def process_pdf(pdf_path):
    """Process a single PDF with enhanced error handling"""
    try:
        text = extract_text(pdf_path)
        if not text:
            print(f"No text extracted from {pdf_path}")
            return []
        
        lines = [l for l in text.split('\n') if len(l.strip()) > 20]
        
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        rows = []
        
        for i, line in enumerate(lines, start=1):
            parsed = robust_parse(line)
            # Add source_pdf, row_in_pdf, raw_text + parsed data
            rows.append([pdf_name, i, line] + parsed)
        
        return rows
    
    except Exception as e:
        print(f"FAILED: {pdf_path} -> {e}")
        return []

# ===================== CHECKPOINTING =====================
def load_checkpoint():
    """Load checkpoint with done files and accumulated rows"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            print("Checkpoint file corrupted, starting fresh")
            return set(), []
    return set(), []

def save_checkpoint(done_files, all_rows):
    """Save checkpoint with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump((done_files, all_rows), f)
    print(f"[{timestamp}] Checkpoint saved: {len(done_files)} files processed")

# ===================== MAIN EXECUTION =====================
if __name__ == '__main__':
    print("=" * 60)
    print("PDF-to-CSV Extraction Started")
    print("=" * 60)
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
    
    # 🔴 LIMIT FOR TESTING - Uncomment to limit
    # pdf_files = pdf_files[:100]
    
    print(f"Total PDFs found: {len(pdf_files)}")
    
    # Load checkpoint
    done_files, all_rows = load_checkpoint()
    remaining = [p for p in pdf_files if p not in done_files]
    
    print(f"Already processed: {len(done_files)}")
    print(f"Remaining PDFs: {len(remaining)}")
    print("=" * 60)
    
    if not remaining:
        print("All files already processed!")
    else:
        # Process with multiprocessing
        pool = mp.Pool(WORKERS)
        
        start_time = datetime.now()
        
        for idx, result in enumerate(pool.imap_unordered(process_pdf, remaining), 1):
            pdf_path = remaining[idx - 1]
            done_files.add(pdf_path)
            all_rows.extend(result)
            
            # Checkpoint every N files
            if idx % CHECKPOINT_EVERY == 0:
                save_checkpoint(done_files, all_rows)
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = idx / elapsed if elapsed > 0 else 0
                remaining_count = len(remaining) - idx
                eta_seconds = remaining_count / rate if rate > 0 else 0
                
                print(f"Progress: {idx}/{len(remaining)} files | "
                      f"Remaining: {remaining_count} | "
                      f"Rate: {rate:.2f} files/sec | "
                      f"ETA: {eta_seconds/60:.1f} min")
        
        pool.close()
        pool.join()
        
        # Final checkpoint
        save_checkpoint(done_files, all_rows)
    
    # Create DataFrame with all 23 columns
    print("=" * 60)
    print("Creating final CSV...")
    master_df = pd.DataFrame(all_rows, columns=target_columns)
    
    # Save to CSV
    output_path = r'C:\pdf2csv\master_output2.csv'
    master_df.to_csv(output_path, index=False)
    
    print(f"Processing complete!")
    print(f"Total rows extracted: {len(master_df)}")
    print(f"CSV saved as: {output_path}")
    print("=" * 60)