import json
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    DATA_PROCESSED_DIR,
    SAVED_MODELS_DIR,
    OUTPUTS_DIR,
    ID_COL,
    TARGET_COL
)
from models import TabularEntityEmbeddingNN
from dataset import FraudTabularDataset
from torch.utils.data import DataLoader


def generate_submission(
    model_checkpoint: str = "nn_focal_vpca.pt",
    metadata_filename: str = "metadata_vpca.json",
    test_parquet_filename: str = "test_vpca.parquet",
    batch_size: int = 4096,
    output_filename: str = "submission.csv"
):
    print("=" * 80)
    print("IEEE-CIS FRAUD DETECTION - KAGGLE SUBMISSION GENERATOR")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Inference Device: {device}")
    
    # 1. Load Metadata
    meta_path = DATA_PROCESSED_DIR / metadata_filename
    print(f"[*] Loading metadata from: {meta_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    cat_cols = meta['categorical_columns']
    num_cols = meta['numerical_columns']
    emb_info = meta['embedding_info']
    
    print(f"[*] Categorical features count: {len(cat_cols)}")
    print(f"[*] Numerical features count: {len(num_cols)}")
    
    # 2. Build Model Architecture & Load State Dict
    print("[*] Instantiating TabularEntityEmbeddingNN architecture...")
    model = TabularEntityEmbeddingNN(
        embedding_info=emb_info,
        cat_cols=cat_cols,
        num_features=len(num_cols),
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
        use_residual=True
    ).to(device)
    
    ckpt_path = SAVED_MODELS_DIR / model_checkpoint
    print(f"[*] Loading trained weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Load Test Data
    test_path = DATA_PROCESSED_DIR / test_parquet_filename
    print(f"[*] Reading preprocessed test set from: {test_path}...")
    t0 = time.time()
    
    # Load required columns only
    cols_to_load = [ID_COL] + cat_cols + num_cols
    test_table = pq.read_table(test_path, columns=cols_to_load)
    df_test = test_table.to_pandas()
    print(f"    Loaded {len(df_test):,} test rows in {time.time() - t0:.2f}s")
    
    test_ids = df_test[ID_COL].values
    
    # 4. Create DataLoader
    test_dataset = FraudTabularDataset(
        df=df_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 5. Batch Inference
    print(f"[*] Running batch inference ({len(test_loader)} batches, batch_size={batch_size})...")
    all_probs = []
    t_inf = time.time()
    
    with torch.no_grad():
        for batch_idx, (cat_x, num_x, _) in enumerate(test_loader, 1):
            cat_x = cat_x.to(device)
            num_x = num_x.to(device)
            logits = model(cat_x, num_x).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            
            if batch_idx % 25 == 0 or batch_idx == len(test_loader):
                elapsed = time.time() - t_inf
                pct = (batch_idx / len(test_loader)) * 100
                print(f"    Progress: {batch_idx:03d}/{len(test_loader)} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s")
                
    all_probs = np.array(all_probs, dtype=np.float32)
    print(f"[*] Inference complete in {time.time() - t_inf:.2f}s")
    
    # 6. Assemble Submission DataFrame
    print("[*] Assembling submission dataframe...")
    sub_df = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: all_probs
    })
    
    # Verification checks
    assert len(sub_df) == 506691, f"Expected 506,691 rows, got {len(sub_df)}"
    assert not sub_df[TARGET_COL].isna().any(), "Submission contains NaN values!"
    assert list(sub_df.columns) == [ID_COL, TARGET_COL], f"Invalid columns: {sub_df.columns}"
    
    # 7. Prediction Distribution Statistics
    print("\n" + "=" * 80)
    print("SUBMISSION PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total Transactions: {len(sub_df):,}")
    print(f"Min Probability:    {sub_df[TARGET_COL].min():.6f}")
    print(f"Max Probability:    {sub_df[TARGET_COL].max():.6f}")
    print(f"Mean Probability:   {sub_df[TARGET_COL].mean():.6f}")
    print(f"Median Probability: {sub_df[TARGET_COL].median():.6f}")
    print(f"Transactions > 0.5: {int((sub_df[TARGET_COL] > 0.5).sum()):,} ({(sub_df[TARGET_COL] > 0.5).mean()*100:.2f}%)")
    print(f"Transactions > 0.3088 (Opt Tau): {int((sub_df[TARGET_COL] > 0.3088).sum()):,} ({(sub_df[TARGET_COL] > 0.3088).mean()*100:.2f}%)")
    print("=" * 80)
    
    # 8. Save CSV and Compressed CSV.GZ
    out_csv = OUTPUTS_DIR / output_filename
    out_gz = OUTPUTS_DIR / f"{output_filename}.gz"
    
    print(f"[*] Saving submission CSV to: {out_csv}")
    sub_df.to_csv(out_csv, index=False)
    
    print(f"[*] Saving compressed GZ to: {out_gz} (Recommended for fast Kaggle upload)")
    sub_df.to_csv(out_gz, index=False, compression='gzip')
    
    csv_size_mb = out_csv.stat().st_size / (1024 * 1024)
    gz_size_mb = out_gz.stat().st_size / (1024 * 1024)
    print(f"    Raw CSV Size:       {csv_size_mb:.2f} MB")
    print(f"    Compressed GZ Size: {gz_size_mb:.2f} MB")
    print("\n[V] SUCCESS: Submission file is 100% ready for Kaggle submission!")
    
    return sub_df


if __name__ == "__main__":
    generate_submission()
