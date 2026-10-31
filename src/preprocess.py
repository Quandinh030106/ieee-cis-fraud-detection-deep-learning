import sys
import os
from pathlib import Path
import json
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_TRANS_PATH,
    TRAIN_IDENT_PATH,
    TEST_TRANS_PATH,
    TEST_IDENT_PATH,
    DATA_PROCESSED_DIR,
    OUTPUTS_DIR,
    OFFICIAL_CAT_TRANS,
    OFFICIAL_CAT_IDENT,
    ALL_CATEGORICAL_FEATURES,
    TRANSACTION_AMT,
    C_COLS,
    D_COLS,
    DIST_COLS,
    ID_NUM_COLS,
    BASE_NUMERICAL_FEATURES,
    TARGET_COL,
    ID_COL,
    TIME_COL,
    TRAIN_VAL_SPLIT_DT,
    calc_embedding_dim,
    RANDOM_SEED
)


def load_v_cluster_info():
    config_path = OUTPUTS_DIR / "v_cluster_config.json"
    pca_trans_path = OUTPUTS_DIR / "v_pca_transformers.pkl"
    if not config_path.exists() or not pca_trans_path.exists():
        raise FileNotFoundError(
            f"V-cluster artifacts missing at {config_path} or {pca_trans_path}. "
            "Please run eda/04_data_mining_contrast.py first!"
        )

    with open(config_path, "r") as f:
        v_cluster_config = json.load(f)

    v_representatives = [v_cluster_config[f"cluster_{i}"]['representative'] for i in range(1, 16)]
    v_pca_models = joblib.load(pca_trans_path)

    return v_cluster_config, v_representatives, v_pca_models


def load_raw_train():
    print("[1/6] Loading raw train transaction & identity datasets...")
    df_trans = pd.read_csv(TRAIN_TRANS_PATH)
    df_ident = pd.read_csv(TRAIN_IDENT_PATH)
    df = pd.merge(df_trans, df_ident, on=ID_COL, how='left')
    print(f"Train merged shape: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    del df_trans, df_ident
    gc.collect()
    return df


def load_raw_test():
    print("[4/6] Loading raw test transaction & identity datasets...")
    df_trans = pd.read_csv(TEST_TRANS_PATH)
    df_ident = pd.read_csv(TEST_IDENT_PATH)

    # In test_identity.csv, column names often have dashes instead of underscores ('id-01' -> 'id_01')
    rename_dict = {col: col.replace('-', '_') for col in df_ident.columns if '-' in col}
    if rename_dict:
        print(f"Renaming {len(rename_dict)} dashed columns in test_identity (e.g. id-01 -> id_01)...")
        df_ident = df_ident.rename(columns=rename_dict)

    df = pd.merge(df_trans, df_ident, on=ID_COL, how='left')
    print(f"Test merged shape: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    del df_trans, df_ident
    gc.collect()
    return df


def build_categorical_encoders(train_df, cat_cols):
    print("[2/6] Building categorical encoders & vocabulary mappings (Train only)...")
    encoders = {}
    embedding_info = {}

    for col in cat_cols:
        # Fill missing with 'Missing' and convert to string
        series = train_df[col].fillna('Missing').astype(str)
        # Unique categories
        unique_vals = list(series.unique())

        # Ensure 'Missing' is index 0
        if 'Missing' in unique_vals:
            unique_vals.remove('Missing')
        unique_vals = ['Missing'] + sorted(unique_vals)

        # Mapping dictionary
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        unseen_idx = len(unique_vals)  # Index for out-of-vocabulary categories in val/test
        num_classes = unseen_idx + 1   # Including __UNSEEN__

        emb_dim = calc_embedding_dim(num_classes)

        encoders[col] = {
            'val_to_idx': val_to_idx,
            'unseen_idx': unseen_idx,
            'num_classes': num_classes,
            'embedding_dim': emb_dim
        }
        embedding_info[col] = {
            'num_classes': num_classes,
            'embedding_dim': emb_dim
        }

    return encoders, embedding_info


def transform_categorical_features(df, encoders, cat_cols):
    df_cat = pd.DataFrame(index=df.index)
    for col in cat_cols:
        enc = encoders[col]
        series = df[col].fillna('Missing').astype(str)
        # Vectorized mapping via map and fillna for unseen values
        df_cat[col] = series.map(enc['val_to_idx']).fillna(enc['unseen_idx']).astype(np.int32)
    return df_cat


def fit_numerical_preprocessors(train_df, base_num_cols, v_reps):
    print("[3/6] Fitting numerical scalers & computing imputation medians (Train only)...")
    train_transformed = pd.DataFrame(index=train_df.index)

    # 1. TransactionAmt -> log1p
    train_transformed[TRANSACTION_AMT] = np.log1p(np.maximum(train_df[TRANSACTION_AMT].fillna(0).to_numpy(), 0))

    # 2. C-cols (counts) -> log1p
    for col in C_COLS:
        if col in train_df.columns:
            train_transformed[col] = np.log1p(np.maximum(train_df[col].fillna(0).to_numpy(), 0))

    # 3. D-cols (intervals) -> clip negative clock drift to 0
    for col in D_COLS:
        if col in train_df.columns:
            train_transformed[col] = np.maximum(train_df[col].to_numpy(), 0)

    # 4. Other base numeric (dist1, dist2, id_01-id_11)
    other_base = DIST_COLS + ID_NUM_COLS
    for col in other_base:
        if col in train_df.columns:
            train_transformed[col] = train_df[col].to_numpy()

    # 5. V15 features
    for col in v_reps:
        if col in train_df.columns:
            train_transformed[col] = train_df[col].to_numpy()

    # Calculate medians for imputation
    medians = {}
    for col in train_transformed.columns:
        med = float(train_transformed[col].median())
        if np.isnan(med):
            med = 0.0
        medians[col] = med
        train_transformed[col] = train_transformed[col].fillna(med)

    # Fit StandardScaler on V15 numerical features
    scaler_v15 = StandardScaler()
    scaler_v15.fit(train_transformed)

    return medians, scaler_v15


def transform_numerical_v15(df, medians, scaler_v15, v_reps):
    cols_order = scaler_v15.feature_names_in_
    df_transformed = pd.DataFrame(index=df.index)

    # 1. TransactionAmt
    df_transformed[TRANSACTION_AMT] = np.log1p(np.maximum(df[TRANSACTION_AMT].fillna(0).to_numpy(), 0))

    # 2. C-cols
    for col in C_COLS:
        if col in df.columns:
            df_transformed[col] = np.log1p(np.maximum(df[col].fillna(0).to_numpy(), 0))
        else:
            df_transformed[col] = 0.0

    # 3. D-cols
    for col in D_COLS:
        if col in df.columns:
            df_transformed[col] = np.maximum(df[col].to_numpy(), 0)
        else:
            df_transformed[col] = np.nan

    # 4. Other base
    for col in (DIST_COLS + ID_NUM_COLS):
        if col in df.columns:
            df_transformed[col] = df[col].to_numpy()
        else:
            df_transformed[col] = np.nan

    # 5. V15
    for col in v_reps:
        if col in df.columns:
            df_transformed[col] = df[col].to_numpy()
        else:
            df_transformed[col] = np.nan

    # Impute using train medians
    for col in cols_order:
        df_transformed[col] = df_transformed[col].fillna(medians[col])

    # Standard scale
    scaled_array = scaler_v15.transform(df_transformed[cols_order])
    scaled_df = pd.DataFrame(scaled_array, columns=cols_order, index=df.index)
    return scaled_df


def transform_pca_v(df, v_pca_models):
    # Transform each cluster using its fitted PCA model
    pca_cols_dict = {}
    for c_idx in range(1, 16):
        c_name = f"cluster_{c_idx}"
        info = v_pca_models[c_name]
        feat_list = info['features']
        scaler = info['scaler']
        pca = info['pca']

        cluster_sub = df[feat_list].copy()
        for f in feat_list:
            if f not in cluster_sub.columns:
                cluster_sub[f] = -1.0
            else:
                cluster_sub[f] = cluster_sub[f].fillna(-1.0)

        scaled = scaler.transform(cluster_sub[feat_list])
        projected = pca.transform(scaled)

        for comp_i in range(projected.shape[1]):
            pca_cols_dict[f"PCA_{c_name}_comp{comp_i + 1}"] = projected[:, comp_i]

    return pd.DataFrame(pca_cols_dict, index=df.index)


def process_and_export():
    print("="*80)
    print("DS201 FULL DATA PREPROCESSING & STANDARDIZATION PIPELINE")
    print("="*80)

    v_cluster_config, v_reps, v_pca_models = load_v_cluster_info()

    # Load Train
    df_train_full = load_raw_train()

    # Temporal split
    is_train_split = df_train_full[TIME_COL] < TRAIN_VAL_SPLIT_DT
    train_indices = df_train_full.index[is_train_split]
    val_indices = df_train_full.index[~is_train_split]

    print(f"Chronological split at TransactionDT={TRAIN_VAL_SPLIT_DT}:")
    print(f"  -> Train split: {len(train_indices):,} records ({len(train_indices)/len(df_train_full)*100:.2f}%)")
    print(f"  -> Val split:   {len(val_indices):,} records ({len(val_indices)/len(df_train_full)*100:.2f}%)")

    # Fit Categorical Encoders on Train split only
    train_split_df = df_train_full.loc[train_indices]
    encoders, embedding_info = build_categorical_encoders(train_split_df, ALL_CATEGORICAL_FEATURES)

    # Fit Numerical Scalers on Train split only
    medians, scaler_v15 = fit_numerical_preprocessors(train_split_df, BASE_NUMERICAL_FEATURES, v_reps)

    # Transform Train & Val Categoricals
    print("\nTransforming Categorical Attributes...")
    cat_transformed_all = transform_categorical_features(df_train_full, encoders, ALL_CATEGORICAL_FEATURES)

    # Transform Train & Val Numerics (V15)
    print("Transforming Numerical Attributes (V15)...")
    num_transformed_v15_all = transform_numerical_v15(df_train_full, medians, scaler_v15, v_reps)

    # Transform Train & Val PCA V-features
    print("Transforming PCA-reduced V-features...")
    v_pca_all = transform_pca_v(df_train_full, v_pca_models)

    # Base numerics excluding V15 for VPCA pipeline
    base_num_cols = [c for c in scaler_v15.feature_names_in_ if c not in v_reps]
    num_transformed_vpca_all = pd.concat([num_transformed_v15_all[base_num_cols], v_pca_all], axis=1)

    # Fit a scaler on VPCA features (Train split only)
    scaler_vpca = StandardScaler()
    scaler_vpca.fit(num_transformed_vpca_all.loc[train_indices])
    scaled_vpca_array = scaler_vpca.transform(num_transformed_vpca_all)
    num_transformed_vpca_all = pd.DataFrame(scaled_vpca_array, columns=scaler_vpca.feature_names_in_, index=df_train_full.index)

    # Build full V15 dataframe
    df_v15_all = pd.concat([
        df_train_full[[ID_COL, TIME_COL, TARGET_COL]],
        cat_transformed_all,
        num_transformed_v15_all
    ], axis=1)

    # Build full VPCA dataframe
    df_vpca_all = pd.concat([
        df_train_full[[ID_COL, TIME_COL, TARGET_COL]],
        cat_transformed_all,
        num_transformed_vpca_all
    ], axis=1)

    # Split into Train and Val
    train_v15 = df_v15_all.loc[train_indices]
    val_v15 = df_v15_all.loc[val_indices]

    train_vpca = df_vpca_all.loc[train_indices]
    val_vpca = df_vpca_all.loc[val_indices]

    # Export Train and Val Parquet files
    print("\n[5/6] Exporting Train and Val Parquet files...")
    train_v15.to_parquet(DATA_PROCESSED_DIR / "train_v15.parquet", index=False)
    val_v15.to_parquet(DATA_PROCESSED_DIR / "val_v15.parquet", index=False)
    train_vpca.to_parquet(DATA_PROCESSED_DIR / "train_vpca.parquet", index=False)
    val_vpca.to_parquet(DATA_PROCESSED_DIR / "val_vpca.parquet", index=False)

    print(f"Exported train_v15.parquet: {train_v15.shape}")
    print(f"Exported val_v15.parquet:   {val_v15.shape}")
    print(f"Exported train_vpca.parquet: {train_vpca.shape}")
    print(f"Exported val_vpca.parquet:   {val_vpca.shape}")

    # Clear memory before processing test set
    del df_train_full, train_split_df, cat_transformed_all, num_transformed_v15_all
    del v_pca_all, num_transformed_vpca_all, df_v15_all, df_vpca_all
    del train_v15, val_v15, train_vpca, val_vpca
    gc.collect()

    # Load Test
    df_test = load_raw_test()

    # Transform Test Categoricals
    print("\nTransforming Test Categoricals...")
    test_cat = transform_categorical_features(df_test, encoders, ALL_CATEGORICAL_FEATURES)

    # Transform Test Numerics (V15)
    print("Transforming Test Numerics (V15)...")
    test_num_v15 = transform_numerical_v15(df_test, medians, scaler_v15, v_reps)

    # Transform Test Numerics (VPCA)
    print("Transforming Test Numerics (VPCA)...")
    test_v_pca = transform_pca_v(df_test, v_pca_models)
    test_num_vpca_raw = pd.concat([test_num_v15[base_num_cols], test_v_pca], axis=1)
    test_num_vpca_scaled = pd.DataFrame(
        scaler_vpca.transform(test_num_vpca_raw[scaler_vpca.feature_names_in_]),
        columns=scaler_vpca.feature_names_in_,
        index=df_test.index
    )

    # Export Test Parquets
    print("\n[6/6] Exporting Test Parquet files...")
    test_v15 = pd.concat([df_test[[ID_COL, TIME_COL]], test_cat, test_num_v15], axis=1)
    test_vpca = pd.concat([df_test[[ID_COL, TIME_COL]], test_cat, test_num_vpca_scaled], axis=1)

    test_v15.to_parquet(DATA_PROCESSED_DIR / "test_v15.parquet", index=False)
    test_vpca.to_parquet(DATA_PROCESSED_DIR / "test_vpca.parquet", index=False)
    print(f"Exported test_v15.parquet:  {test_v15.shape}")
    print(f"Exported test_vpca.parquet: {test_vpca.shape}")

    # Save encoders, scalers, metadata
    joblib.dump(encoders, DATA_PROCESSED_DIR / "label_encoders.pkl")
    joblib.dump(scaler_v15, DATA_PROCESSED_DIR / "scaler_v15.pkl")
    joblib.dump(scaler_vpca, DATA_PROCESSED_DIR / "scaler_vpca.pkl")

    meta_v15 = {
        'num_train_samples': int(len(train_indices)),
        'num_val_samples': int(len(val_indices)),
        'num_test_samples': int(len(df_test)),
        'target_column': TARGET_COL,
        'categorical_columns': ALL_CATEGORICAL_FEATURES,
        'numerical_columns': list(scaler_v15.feature_names_in_),
        'embedding_info': embedding_info,
        'v_representation': '15_cluster_representatives',
        'v_representatives': v_reps
    }
    with open(DATA_PROCESSED_DIR / "metadata_v15.json", "w") as f:
        json.dump(meta_v15, f, indent=2)

    meta_vpca = {
        'num_train_samples': int(len(train_indices)),
        'num_val_samples': int(len(val_indices)),
        'num_test_samples': int(len(df_test)),
        'target_column': TARGET_COL,
        'categorical_columns': ALL_CATEGORICAL_FEATURES,
        'numerical_columns': list(scaler_vpca.feature_names_in_),
        'embedding_info': embedding_info,
        'v_representation': 'cluster_wise_pca_orthogonal',
        'pca_components_per_cluster': {k: int(v['pca'].n_components_) for k, v in v_pca_models.items()}
    }
    with open(DATA_PROCESSED_DIR / "metadata_vpca.json", "w") as f:
        json.dump(meta_vpca, f, indent=2)

    print("\n[SUCCESS] All preprocessed datasets, scalers, and metadata successfully exported!")


if __name__ == "__main__":
    process_and_export()
