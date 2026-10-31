import pandas as pd
import numpy as np
import os
import gc

NROWS = None
DATA_DIR = os.path.join("data", "raw", "ieee-fraud-detection")
PATH_TRANS = os.path.join(DATA_DIR, "train_transaction.csv")
PATH_IDENT = os.path.join(DATA_DIR, "train_identity.csv")


def load_and_merge_data(path_trans, path_ident, nrows=None):
    df_trans = pd.read_csv(path_trans, nrows=nrows)
    df_ident = pd.read_csv(path_ident, nrows=nrows)
    
    print(f" - Bảng Transaction: {df_trans.shape[0]} dòng, {df_trans.shape[1]} cột")
    print(f" - Bảng Identity:    {df_ident.shape[0]} dòng, {df_ident.shape[1]} cột")
    
    # Ghép bảng theo TransactionID
    df = pd.merge(df_trans, df_ident, on="TransactionID", how="left")
    print(f"Bảng sau khi ghép: {df.shape[0]} dòng, {df.shape[1]} cột")

    del df_trans, df_ident
    gc.collect()
    
    return df


def scan_pseudo_missing_strings(df):
    print("\n" + "="*70)
    
    # Danh sách các từ khóa thường bị dùng thay cho missing
    suspect_words = {
        "unknown", "other", "none", "null", "nan", "na", 
        "missing", "not found", "undefined", "?", "-", ""
    }
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"Tổng số cột dạng phân loại/chuỗi: {len(cat_cols)}")
    
    found_count = 0
    for col in cat_cols:
        # Lấy các giá trị độc nhất
        unique_vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
        matched = set(unique_vals).intersection(suspect_words)
        
        # Kiểm tra thêm trường hợp chuỗi chỉ chứa khoảng trắng rỗng
        has_blank = any(val == "" for val in unique_vals)
        if has_blank:
            matched.add("<chuỗi rỗng>")
            
        if matched:
            found_count += 1
            # Đếm số lượng cụ thể của từng từ khóa xuất hiện trong cột
            sample_counts = {}
            for w in matched:
                if w == "<chuỗi rỗng>":
                    cnt = (df[col].astype(str).str.strip() == "").sum()
                else:
                    cnt = (df[col].astype(str).str.strip().str.lower() == w).sum()
                sample_counts[w] = cnt
                
            print(f" - Cột [{col}]: chứa các giá trị nghi vấn -> {sample_counts}")
            
    if found_count == 0:
        print("Không tìm thấy từ khóa nghi vấn rõ ràng nào trong các cột chữ.")


def scan_numeric_sentinels(df):
    print("\n" + "="*70)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Bỏ qua các cột ID và nhãn mục tiêu
    ignore_cols = {"TransactionID", "isFraud"}
    scan_cols = [c for c in num_cols if c not in ignore_cols]
    
    print(f"Tổng số cột dạng số cần kiểm tra: {len(scan_cols)}")
    
    found_negatives = []
    found_sentinels = []
    
    # Các giá trị số hay được dùng làm cờ đánh dấu lỗi/thiếu
    common_sentinels = {-1, -99, -999, 999, 9999}
    
    for col in scan_cols:
        col_series = df[col].dropna()
        if len(col_series) == 0:
            continue
            
        # Kiểm tra xem có giá trị âm không (đặc biệt quan trọng với các cột tiền, đếm, khoảng cách)
        min_val = col_series.min()
        if min_val < 0:
            count_neg = (col_series < 0).sum()
            found_negatives.append((col, min_val, count_neg))
            
        # Kiểm tra xem có chứa đúng các số sentinel cụ thể không
        val_set = set(col_series.unique())
        matched_sentinels = val_set.intersection(common_sentinels)
        if matched_sentinels:
            details = {s: (col_series == s).sum() for s in matched_sentinels}
            found_sentinels.append((col, details))
            
    # In kết quả các cột có số âm
    print(f"\nSố cột có xuất hiện GIÁ TRỊ ÂM: {len(found_negatives)}")
    for col, min_v, cnt in found_negatives[:10]: # in mẫu 10 cột đầu tiên
        print(f"Cột [{col}]: Giá trị nhỏ nhất = {min_v}, có {cnt} dòng bị âm")
    if len(found_negatives) > 10:
        print(f"và còn {len(found_negatives) - 10} cột khác có giá trị âm.")
        
    # In kết quả các cột có số sentinel
    print(f"\nSố cột có chứa các số quy ước (-1, -999, 9999,...): {len(found_sentinels)}")
    for col, details in found_sentinels[:10]:
        print(f"Cột [{col}]: {details}")
    if len(found_sentinels) > 10:
        print(f"và còn {len(found_sentinels) - 10} cột khác có số sentinel.")


def scan_mixed_data_types(df):
    print("\n" + "="*70)

    cat_cols = df.select_dtypes(include=["object"]).columns
    mixed_cols = []
    
    for col in cat_cols:
        col_series = df[col].dropna()
        if len(col_series) == 0:
            continue
            
        # Thử ép sang số bằng pd.to_numeric
        # errors='coerce' sẽ biến các giá trị chữ không ép được thành NaN
        converted = pd.to_numeric(col_series, errors='coerce')
        numeric_count = converted.notna().sum()
        total_non_null = len(col_series)
        
        # Nếu có trên 50% dữ liệu là số nhưng vẫn bị lưu dạng object -> Nghi vấn lẫn chữ
        if numeric_count > 0 and numeric_count < total_non_null:
            pct_num = (numeric_count / total_non_null) * 100
            mixed_cols.append((col, pct_num, total_non_null - numeric_count))
            
    print(f"Số cột nghi ngờ bị lẫn lộn giữa SỐ và CHỮ: {len(mixed_cols)}")
    for col, pct, str_cnt in mixed_cols:
        print(f"Cột [{col}]: {pct:.1f}% là số, nhưng có {str_cnt} dòng là chữ (khiến cột bị ép thành object)")


def summarize_missing_reality(df):
    print("\n" + "="*70)
    
    missing_pct = (df.isna().sum() / len(df)) * 100
    
    c_clean = (missing_pct == 0).sum()
    c_low = ((missing_pct > 0) & (missing_pct <= 20)).sum()
    c_mid = ((missing_pct > 20) & (missing_pct <= 70)).sum()
    c_high = (missing_pct > 70).sum()
    
    print(f"Tổng số cột phân tích: {df.shape[1]}")
    print(f"Số cột SẠCH 100% (không có NaN nào): {c_clean} cột")
    print(f"Số cột missing nhẹ (<= 20%):          {c_low} cột")
    print(f"Số cột missing vừa (20% - 70%):       {c_mid} cột")
    print(f"Số cột missing NẶNG (> 70%):          {c_high} cột")

if __name__ == "__main__":
    # 1. Nạp và ghép bảng
    df = load_and_merge_data(PATH_TRANS, PATH_IDENT, nrows=NROWS)
    
    # 2. Quét missing ngầm dạng chữ
    scan_pseudo_missing_strings(df)
    
    # 3. Quét số sentinel dạng số
    scan_numeric_sentinels(df)
    
    # 4. Quét kiểu dữ liệu hỗn hợp
    scan_mixed_data_types(df)
    
    # 5. Tổng kết missing thật
    summarize_missing_reality(df)
    
    print("\n[*] Hoàn tất kiểm tra dữ liệu!")