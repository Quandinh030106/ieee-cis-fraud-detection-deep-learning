import sys
import os
from pathlib import Path

# Ensure unbuffered output and writable matplotlib cache
sys.stdout.reconfigure(line_buffering=True)
mpl_dir = Path(__file__).resolve().parent.parent / "outputs" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIR,
    SAVED_MODELS_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    RANDOM_SEED,
    FOCAL_ALPHA,
    FOCAL_GAMMA
)
from src.dataset import create_dataloader
from src.models import TabularEntityEmbeddingNN
from src.losses import BinaryFocalLoss
from src.train import train_model, evaluate_epoch
from src.evaluate import evaluate_predictions


def run_all_benchmarks():
    print("="*80)
    print("DS201 FRAUD DETECTION - DEEP LEARNING BENCHMARK EXPERIMENTS")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load Metadata & Parquet Datasets
    print("\n[Step 1] Loading preprocessed datasets and metadata...")
    with open(DATA_PROCESSED_DIR / "metadata_v15.json", "r") as f:
        meta_v15 = json.load(f)

    with open(DATA_PROCESSED_DIR / "metadata_vpca.json", "r") as f:
        meta_vpca = json.load(f)

    df_train_v15 = pd.read_parquet(DATA_PROCESSED_DIR / "train_v15.parquet")
    df_val_v15 = pd.read_parquet(DATA_PROCESSED_DIR / "val_v15.parquet")

    df_train_vpca = pd.read_parquet(DATA_PROCESSED_DIR / "train_vpca.parquet")
    df_val_vpca = pd.read_parquet(DATA_PROCESSED_DIR / "val_vpca.parquet")

    cat_cols = meta_v15['categorical_columns']
    num_cols_v15 = meta_v15['numerical_columns']
    num_cols_vpca = meta_vpca['numerical_columns']
    target_col = meta_v15['target_column']
    emb_info = meta_v15['embedding_info']

    print(f"Data Loaded: Train = {len(df_train_v15):,} rows | Val = {len(df_val_v15):,} rows")
    print(f"Features: 49 Categorical, {len(num_cols_v15)} Numeric (V15), {len(num_cols_vpca)} Numeric (VPCA)")

    results_list = []

    # -------------------------------------------------------------------------
    # Experiment 1: Baseline Logistic Regression (Numerical V15 only)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 1: Baseline Logistic Regression (Numerical V15 Features)")
    print("-"*80)
    X_train_num = df_train_v15[num_cols_v15].to_numpy()
    y_train = df_train_v15[target_col].to_numpy()
    X_val_num = df_val_v15[num_cols_v15].to_numpy()
    y_val = df_val_v15[target_col].to_numpy()

    lr_model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, class_weight='balanced')
    lr_model.fit(X_train_num, y_train)
    lr_probs = lr_model.predict_proba(X_val_num)[:, 1]

    metrics_lr = evaluate_predictions(
        y_val, lr_probs, save_dir=FIGURES_DIR, model_name="Logistic_Regression_V15"
    )
    opt_lr = metrics_lr['optimal_metrics']
    print(f"LR Baseline -> Val ROC-AUC: {metrics_lr['roc_auc']:.4f} | PR-AUC: {metrics_lr['pr_auc']:.4f}")
    print(f"  -> Optimal Threshold: {metrics_lr['optimal_threshold']:.4f} | F1: {opt_lr['f1_score']:.4f} | Precision: {opt_lr['precision']:.4f} | Recall: {opt_lr['recall']:.4f}")
    print(f"  -> Confusion Matrix: TP={opt_lr['tp']}, FP={opt_lr['fp']}, FN={opt_lr['fn']}, TN={opt_lr['tn']}")
    results_list.append({
        'Model': 'Logistic Regression (Baseline)',
        'Feature_Space': 'V15 Numerical Only',
        'Loss_Function': 'Weighted Cross-Entropy',
        'ROC_AUC': round(metrics_lr['roc_auc'], 4),
        'PR_AUC': round(metrics_lr['pr_auc'], 4),
        'F1_Score': round(opt_lr['f1_score'], 4),
        'Precision': round(opt_lr['precision'], 4),
        'Recall': round(opt_lr['recall'], 4),
        'TP': opt_lr['tp'],
        'FP': opt_lr['fp'],
        'FN': opt_lr['fn'],
        'TN': opt_lr['tn'],
        'Optimal_Threshold': round(metrics_lr['optimal_threshold'], 4)
    })

    # -------------------------------------------------------------------------
    # Experiment 2: Deep Entity Embedding NN with Standard BCE Loss (V15)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 2: Deep Entity Embedding NN with Standard BCE (V15)")
    print("-"*80)
    train_loader_v15 = create_dataloader(df_train_v15, cat_cols, num_cols_v15, target_col, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_v15 = create_dataloader(df_val_v15, cat_cols, num_cols_v15, target_col, batch_size=BATCH_SIZE, shuffle=False)

    model_bce = TabularEntityEmbeddingNN(
        embedding_info=emb_info,
        cat_cols=cat_cols,
        num_features=len(num_cols_v15),
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
        use_residual=True
    ).to(device)

    # Standard BCE with positive class weight
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer_bce = torch.optim.AdamW(model_bce.parameters(), lr=1e-3, weight_decay=1e-4)

    #add scheduler
    scheduler_bce = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_bce, mode='min', factor=0.5, patience=3
)

    trained_bce = train_model(
        model=model_bce,
        train_loader=train_loader_v15,
        val_loader=val_loader_v15,
        criterion=criterion_bce,
        optimizer=optimizer_bce,
        scheduler=scheduler_bce,
        max_epochs=50,
        patience=7,     #Change number of epochs and patience to see if the accuracy improves
        device=device,
        save_path=SAVED_MODELS_DIR / "nn_bce_v15.pt"
    )

    _, _, _, val_probs_bce = evaluate_epoch(model_bce, val_loader_v15, criterion_bce, device)
    metrics_bce = evaluate_predictions(
        y_val, val_probs_bce, save_dir=FIGURES_DIR, model_name="Deep_Entity_NN_BCE_V15"
    )
    opt_bce = metrics_bce['optimal_metrics']
    print(f"Deep NN (BCE) -> Val ROC-AUC: {metrics_bce['roc_auc']:.4f} | PR-AUC: {metrics_bce['pr_auc']:.4f}")
    print(f"  -> Optimal Threshold: {metrics_bce['optimal_threshold']:.4f} | F1: {opt_bce['f1_score']:.4f} | Precision: {opt_bce['precision']:.4f} | Recall: {opt_bce['recall']:.4f}")
    print(f"  -> Confusion Matrix: TP={opt_bce['tp']}, FP={opt_bce['fp']}, FN={opt_bce['fn']}, TN={opt_bce['tn']}")
    results_list.append({
        'Model': 'Tabular Entity-Embedding NN',
        'Feature_Space': '49 Cats + V15',
        'Loss_Function': 'Weighted BCE',
        'ROC_AUC': round(metrics_bce['roc_auc'], 4),
        'PR_AUC': round(metrics_bce['pr_auc'], 4),
        'F1_Score': round(opt_bce['f1_score'], 4),
        'Precision': round(opt_bce['precision'], 4),
        'Recall': round(opt_bce['recall'], 4),
        'TP': opt_bce['tp'],
        'FP': opt_bce['fp'],
        'FN': opt_bce['fn'],
        'TN': opt_bce['tn'],
        'Optimal_Threshold': round(metrics_bce['optimal_threshold'], 4)
    })

    # -------------------------------------------------------------------------
    # Experiment 3: Deep Entity Embedding NN with Focal Loss (V15)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 3: Deep Entity Embedding NN with Focal Loss (V15)")
    print("-"*80)
    model_focal_v15 = TabularEntityEmbeddingNN(
        embedding_info=emb_info,
        cat_cols=cat_cols,
        num_features=len(num_cols_v15),
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
        use_residual=True
    ).to(device)

    criterion_focal = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer_focal = torch.optim.AdamW(model_focal_v15.parameters(), lr=1e-3, weight_decay=1e-4)
    
    #add scheduler
    scheduler_focal = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_focal, mode='min', factor=0.5, patience=3
)

    trained_focal_v15 = train_model(
        model=model_focal_v15,
        train_loader=train_loader_v15,
        val_loader=val_loader_v15,
        criterion=criterion_focal,
        optimizer=optimizer_focal,
        scheduler=scheduler_focal,
        max_epochs=50,
        patience=7,
        device=device,
        save_path=SAVED_MODELS_DIR / "nn_focal_v15.pt"
    )

    _, _, _, val_probs_focal_v15 = evaluate_epoch(model_focal_v15, val_loader_v15, criterion_focal, device)
    metrics_focal_v15 = evaluate_predictions(
        y_val, val_probs_focal_v15, save_dir=FIGURES_DIR, model_name="Deep_Entity_NN_Focal_V15"
    )
    opt_f15 = metrics_focal_v15['optimal_metrics']
    print(f"Deep NN (Focal V15) -> Val ROC-AUC: {metrics_focal_v15['roc_auc']:.4f} | PR-AUC: {metrics_focal_v15['pr_auc']:.4f}")
    print(f"  -> Optimal Threshold: {metrics_focal_v15['optimal_threshold']:.4f} | F1: {opt_f15['f1_score']:.4f} | Precision: {opt_f15['precision']:.4f} | Recall: {opt_f15['recall']:.4f}")
    print(f"  -> Confusion Matrix: TP={opt_f15['tp']}, FP={opt_f15['fp']}, FN={opt_f15['fn']}, TN={opt_f15['tn']}")
    results_list.append({
        'Model': 'Tabular Entity-Embedding NN',
        'Feature_Space': '49 Cats + V15',
        'Loss_Function': f'Focal Loss (a={FOCAL_ALPHA}, g={FOCAL_GAMMA})',
        'ROC_AUC': round(metrics_focal_v15['roc_auc'], 4),
        'PR_AUC': round(metrics_focal_v15['pr_auc'], 4),
        'F1_Score': round(opt_f15['f1_score'], 4),
        'Precision': round(opt_f15['precision'], 4),
        'Recall': round(opt_f15['recall'], 4),
        'TP': opt_f15['tp'],
        'FP': opt_f15['fp'],
        'FN': opt_f15['fn'],
        'TN': opt_f15['tn'],
        'Optimal_Threshold': round(metrics_focal_v15['optimal_threshold'], 4)
    })

    # -------------------------------------------------------------------------
    # Experiment 4: Deep Entity Embedding NN with Focal Loss (VPCA)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 4: Deep Entity Embedding NN with Focal Loss (Cluster-Wise VPCA)")
    print("-"*80)
    train_loader_vpca = create_dataloader(df_train_vpca, cat_cols, num_cols_vpca, target_col, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_vpca = create_dataloader(df_val_vpca, cat_cols, num_cols_vpca, target_col, batch_size=BATCH_SIZE, shuffle=False)

    model_focal_vpca = TabularEntityEmbeddingNN(
        embedding_info=emb_info,
        cat_cols=cat_cols,
        num_features=len(num_cols_vpca),
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
        use_residual=True
    ).to(device)

    optimizer_focal_vpca = torch.optim.AdamW(model_focal_vpca.parameters(), lr=1e-3, weight_decay=1e-4)

     #add scheduler
    scheduler_focal_vpca = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_focal_vpca, mode='min', factor=0.5, patience=3
)


    trained_focal_vpca = train_model(
        model=model_focal_vpca,
        train_loader=train_loader_vpca,
        val_loader=val_loader_vpca,
        criterion=criterion_focal,
        optimizer=optimizer_focal_vpca,
        scheduler=scheduler_focal_vpca,
        max_epochs=50,
        patience=7,
        device=device,
        save_path=SAVED_MODELS_DIR / "nn_focal_vpca.pt"
    )

    _, _, _, val_probs_focal_vpca = evaluate_epoch(model_focal_vpca, val_loader_vpca, criterion_focal, device)
    metrics_focal_vpca = evaluate_predictions(
        y_val, val_probs_focal_vpca, save_dir=FIGURES_DIR, model_name="Deep_Entity_NN_Focal_VPCA"
    )
    opt_fpca = metrics_focal_vpca['optimal_metrics']
    print(f"Deep NN (Focal VPCA) -> Val ROC-AUC: {metrics_focal_vpca['roc_auc']:.4f} | PR-AUC: {metrics_focal_vpca['pr_auc']:.4f}")
    print(f"  -> Optimal Threshold: {metrics_focal_vpca['optimal_threshold']:.4f} | F1: {opt_fpca['f1_score']:.4f} | Precision: {opt_fpca['precision']:.4f} | Recall: {opt_fpca['recall']:.4f}")
    print(f"  -> Confusion Matrix: TP={opt_fpca['tp']}, FP={opt_fpca['fp']}, FN={opt_fpca['fn']}, TN={opt_fpca['tn']}")
    results_list.append({
        'Model': 'Tabular Entity-Embedding NN',
        'Feature_Space': '49 Cats + Orthogonal VPCA',
        'Loss_Function': f'Focal Loss (a={FOCAL_ALPHA}, g={FOCAL_GAMMA})',
        'ROC_AUC': round(metrics_focal_vpca['roc_auc'], 4),
        'PR_AUC': round(metrics_focal_vpca['pr_auc'], 4),
        'F1_Score': round(opt_fpca['f1_score'], 4),
        'Precision': round(opt_fpca['precision'], 4),
        'Recall': round(opt_fpca['recall'], 4),
        'TP': opt_fpca['tp'],
        'FP': opt_fpca['fp'],
        'FN': opt_fpca['fn'],
        'TN': opt_fpca['tn'],
        'Optimal_Threshold': round(metrics_focal_vpca['optimal_threshold'], 4)
    })

    # -------------------------------------------------------------------------
    # Save & Summarize Benchmarks
    # -------------------------------------------------------------------------
    res_df = pd.DataFrame(results_list)
    csv_path = OUTPUTS_DIR / "benchmark_results.csv"
    res_df.to_csv(csv_path, index=False)
    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("="*80)
    print(res_df.to_string(index=False))
    print(f"\nSaved benchmark comparison table to {csv_path}")

    # Export LaTeX table format for paper
    latex_table = res_df.to_latex(index=False)
    with open(OUTPUTS_DIR / "benchmark_table.tex", "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX formatted table to {OUTPUTS_DIR / 'benchmark_table.tex'}")


if __name__ == "__main__":
    run_all_benchmarks()
