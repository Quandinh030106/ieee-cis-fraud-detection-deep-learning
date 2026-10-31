import pandas as pd
import os
import sys

DATA_DIR = os.path.join("data", "raw", "ieee-fraud-detection")
PATH_TRANS = os.path.join(DATA_DIR, "train_transaction.csv")
PATH_IDENT = os.path.join(DATA_DIR, "train_identity.csv")

OFFICIAL_CAT_TRANS = [
    'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2',
    'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

OFFICIAL_CAT_IDENT = [
    'DeviceType', 'DeviceInfo',
    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
    'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
    'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35',
    'id_36', 'id_37', 'id_38'
]


def inspect_official_49_categorical_columns():
    if not os.path.exists(PATH_TRANS) or not os.path.exists(PATH_IDENT):
        print("    Vui lòng kiểm tra lại thư mục chạy lệnh.")
        return

    print(f" - Bảng Transaction: {len(OFFICIAL_CAT_TRANS)} cột")
    print(f" - Bảng Identity:    {len(OFFICIAL_CAT_IDENT)} cột")

    # 1. Chỉ đọc các cột phân loại chính thức + TransactionID
    print("Đang đọc các cột phân loại từ Transaction...")
    df_trans = pd.read_csv(PATH_TRANS, usecols=['TransactionID'] + OFFICIAL_CAT_TRANS)
    
    print("Đang đọc các cột phân loại từ Identity...")
    df_ident = pd.read_csv(PATH_IDENT, usecols=['TransactionID'] + OFFICIAL_CAT_IDENT)

    # 2. Ghép 2 bảng bằng LEFT JOIN
    df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')

    all_cat_cols = OFFICIAL_CAT_TRANS + OFFICIAL_CAT_IDENT
    total_rows = len(df)
    print(f"Ghép thành công. Tổng số dòng: {total_rows:,}, Số cột phân loại: {len(all_cat_cols)}\n")

    # 3. VÒNG LẶP TỪNG CỘT
    for i, col in enumerate(all_cat_cols, 1):
        missing_cnt = df[col].isna().sum()
        missing_pct = (missing_cnt / total_rows) * 100
        n_unique = df[col].nunique(dropna=True)
        raw_dtype = df[col].dtype

        print("=" * 80)
        print(f"[{i:02d}/{len(all_cat_cols)}] Cột: [{col}]  |  Kiểu lưu trữ gốc: {raw_dtype}")
        print(f" - Số nhóm giá trị khác biệt (Unique): {n_unique:,}")
        print(f" - Tỷ lệ Missing (NaN): {missing_cnt:,} dòng ({missing_pct:.2f}%)")

        # Đếm tần suất xuất hiện của từng giá trị (kể cả NaN)
        val_counts = df[col].value_counts(dropna=False)

        # TRƯỜNG HỢP 1: Cột có ít nhóm (<= 30 nhóm) -> In ra toàn bộ 100%
        if n_unique <= 30:
            print(" - TOÀN BỘ CÁC GIÁ TRỊ VÀ TẦN SUẤT:")
            for val, count in val_counts.items():
                pct = (count / total_rows) * 100
                print(f"     * {repr(val):<30}: {count:>8,} dòng ({pct:6.2f}%)")
                
        # TRƯỜNG HỢP 2: Cột có nhiều nhóm (> 30 nhóm, như card1, addr1, email, device...)
        else:
            print(f" - Cột có {n_unique:,} nhóm giá trị khác nhau. Trích xuất mẫu:")
            print("   [Top 5 phổ biến nhất]:")
            for val, count in val_counts.head(5).items():
                pct = (count / total_rows) * 100
                print(f"     * {repr(val):<30}: {count:>8,} dòng ({pct:6.2f}%)")
            print("   [Bottom 5 ít gặp nhất - Các nhóm hiếm gặp / Outlier]:")
            for val, count in val_counts.tail(5).items():
                pct = (count / total_rows) * 100
                print(f"     * {repr(val):<30}: {count:>8,} dòng ({pct:6.2f}%)")


if __name__ == "__main__":
    inspect_official_49_categorical_columns()