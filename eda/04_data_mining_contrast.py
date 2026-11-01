import sys
import os
from pathlib import Path

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Set writable MPLCONFIGDIR inside workspace before importing matplotlib
workspace_dir = Path(__file__).resolve().parent.parent
mpl_dir = workspace_dir / "outputs" / ".matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure parent directory is on sys.path
sys.path.append(str(workspace_dir))
from src.config import (
    TRAIN_TRANS_PATH,
    TRAIN_IDENT_PATH,
    OUTPUTS_DIR,
    FIGURES_DIR,
    OFFICIAL_CAT_TRANS,
    OFFICIAL_CAT_IDENT,
    ALL_CATEGORICAL_FEATURES,
    V_COLS,
    C_COLS,
    D_COLS,
    DIST_COLS,
    ID_NUM_COLS,
    TRANSACTION_AMT,
    TARGET_COL,
    TIME_COL,
    TRAIN_VAL_SPLIT_DT,
    RANDOM_SEED
)


def load_dataset():
    print("[1/5] Loading train transaction and identity data...", flush=True)
    if not TRAIN_TRANS_PATH.exists() or not TRAIN_IDENT_PATH.exists():
        raise FileNotFoundError(
            f"\n\n[!] KHÔNG TÌM THẤY DỮ LIỆU THÔ TẠI: {TRAIN_TRANS_PATH}\n"
            "--------------------------------------------------------------------------------\n"
            "NGUYÊN NHÂN:\n"
            "Các file CSV thô của Kaggle (train_transaction.csv nặng ~683MB) vượt quá giới hạn\n"
            "100MB của GitHub nên đã được đưa vào .gitignore và KHÔNG có sẵn khi clone repo.\n\n"
            "CÁCH KHẮC PHỤC:\n"
            "1. Tải bộ dữ liệu từ Kaggle IEEE-CIS Fraud Detection:\n"
            "   https://www.kaggle.com/c/ieee-fraud-detection/data\n"
            "2. Giải nén các file CSV vào đúng thư mục:\n"
            f"   {TRAIN_TRANS_PATH.parent}\n\n"
            "LƯU Ý QUAN TRỌNG:\n"
            "Toàn bộ kết quả thống kê và đồ thị chẩn đoán ĐÃ CÓ SẴN trong thư mục 'outputs/'!\n"
            "Dữ liệu đã qua tiền xử lý cũng ĐÃ CÓ SẴN trong thư mục 'data/processed/' dạng Parquet!\n"
            "Nếu bạn muốn huấn luyện mô hình, hãy chạy thẳng:\n"
            "   python src/run_experiments.py\n"
            "--------------------------------------------------------------------------------\n"
        )
    df_trans = pd.read_csv(TRAIN_TRANS_PATH)
    df_ident = pd.read_csv(TRAIN_IDENT_PATH)
    df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
    print(f"Loaded merged data: {df.shape[0]:,} rows, {df.shape[1]} columns.", flush=True)
    return df


def calculate_ks_statistics(df, num_cols, target_col):
    print("\n[2/5] Computing Kolmogorov-Smirnov (KS) statistics for numerical features...", flush=True)
    ks_results = []
    fraud_mask = df[target_col] == 1
    legit_mask = df[target_col] == 0

    for col in num_cols:
        if col not in df.columns:
            continue
        s_legit = df.loc[legit_mask, col].dropna()
        s_fraud = df.loc[fraud_mask, col].dropna()

        if len(s_legit) > 10 and len(s_fraud) > 10:
            stat, p_val = ks_2samp(s_legit, s_fraud)
            ks_results.append({
                'feature': col,
                'ks_statistic': float(stat),
                'p_value': float(p_val),
                'legit_count': int(len(s_legit)),
                'fraud_count': int(len(s_fraud)),
                'legit_median': float(s_legit.median()),
                'fraud_median': float(s_fraud.median())
            })

    ks_df = pd.DataFrame(ks_results).sort_values(by='ks_statistic', ascending=False).reset_index(drop=True)
    out_path = OUTPUTS_DIR / "ks_test_results.csv"
    ks_df.to_csv(out_path, index=False)
    print(f"Saved KS test results to {out_path} ({len(ks_df)} features ranked). Top 5 features:", flush=True)
    for idx, row in ks_df.head(5).iterrows():
        print(f"  {row['feature']:<12}: KS = {row['ks_statistic']:.4f} (p = {row['p_value']:.2e})", flush=True)
    return ks_df


def calculate_information_value(df, cat_cols, target_col, max_bins=20):
    print("\n[3/5] Computing Weight of Evidence (WoE) and Information Value (IV) for categorical features...", flush=True)
    iv_results = []
    total_fraud = (df[target_col] == 1).sum()
    total_legit = (df[target_col] == 0).sum()

    for col in cat_cols:
        if col not in df.columns:
            continue
        # Convert nulls to explicit string 'Missing'
        s = df[col].fillna('Missing').astype(str)
        # Group rare categories if cardinality is extremely high
        val_counts = s.value_counts()
        if len(val_counts) > max_bins:
            top_cats = set(val_counts.head(max_bins - 1).index)
            s = s.apply(lambda x: x if x in top_cats else 'Other_Grouped')

        grouped = df.groupby(s, observed=False)[target_col].agg(['count', 'sum']).reset_index()
        grouped.columns = ['bin', 'total', 'fraud']
        grouped['legit'] = grouped['total'] - grouped['fraud']

        # Laplace smoothing to avoid division by zero
        grouped['dist_fraud'] = (grouped['fraud'] + 0.5) / (total_fraud + 1.0)
        grouped['dist_legit'] = (grouped['legit'] + 0.5) / (total_legit + 1.0)
        grouped['woe'] = np.log(grouped['dist_fraud'] / grouped['dist_legit'])
        grouped['iv'] = (grouped['dist_fraud'] - grouped['dist_legit']) * grouped['woe']

        total_iv = grouped['iv'].sum()
        iv_results.append({
            'feature': col,
            'information_value': float(total_iv),
            'unique_bins': int(len(grouped)),
            'pred_power': (
                'Suspicious/Overfitting' if total_iv > 0.5 else
                'Strong' if total_iv >= 0.3 else
                'Medium' if total_iv >= 0.1 else
                'Weak' if total_iv >= 0.02 else 'Useless'
            )
        })

    iv_df = pd.DataFrame(iv_results).sort_values(by='information_value', ascending=False).reset_index(drop=True)
    out_path = OUTPUTS_DIR / "iv_results.csv"
    iv_df.to_csv(out_path, index=False)
    print(f"Saved IV results to {out_path} ({len(iv_df)} features ranked). Top 5 features:", flush=True)
    for idx, row in iv_df.head(5).iterrows():
        print(f"  {row['feature']:<15}: IV = {row['information_value']:.4f} ({row['pred_power']})", flush=True)
    return iv_df


def analyze_v_clusters_and_pca(df, v_cols, ks_df):
    print("\n[4/5] Grouping 339 V-columns into 15 null-pattern clusters & computing PCA...", flush=True)
    null_patterns = {}
    for col in v_cols:
        if col in df.columns:
            null_mask = df[col].isna().to_numpy()
            null_count = int(null_mask.sum())
            null_patterns[col] = (null_count, null_mask)

    clusters = []
    visited = set()

    for col_a, (count_a, mask_a) in null_patterns.items():
        if col_a in visited:
            continue
        cluster = [col_a]
        visited.add(col_a)
        for col_b, (count_b, mask_b) in null_patterns.items():
            if col_b in visited:
                continue
            if count_a == count_b and np.array_equal(mask_a, mask_b):
                cluster.append(col_b)
                visited.add(col_b)
        clusters.append(cluster)

    clusters = sorted(clusters, key=lambda c: int(c[0].replace('V', '')))
    print(f"Found exactly {len(clusters)} V-clusters matching null masks.", flush=True)

    ks_dict = dict(zip(ks_df['feature'], ks_df['ks_statistic']))

    v_cluster_config = {}
    summary_rows = []
    pca_transformers = {}
    pca_dimensions = {}

    for c_idx, cluster in enumerate(clusters, 1):
        best_feat = max(cluster, key=lambda col: ks_dict.get(col, 0.0))
        missing_rate = float(df[best_feat].isna().mean() * 100)

        indices = [int(c.replace('V', '')) for c in cluster]
        v_range = f"V{min(indices)}-V{max(indices)}"

        cluster_data = df[cluster].fillna(-1.0)
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(cluster_data)

        pca = PCA(n_components=min(len(cluster), 3), random_state=RANDOM_SEED)
        pca.fit(scaled_vals)
        n_comps = int(pca.n_components_)

        pca_transformers[f"cluster_{c_idx}"] = {
            'features': cluster,
            'scaler': scaler,
            'pca': pca
        }
        pca_dimensions[f"cluster_{c_idx}"] = n_comps

        v_cluster_config[f"cluster_{c_idx}"] = {
            'features': cluster,
            'representative': best_feat,
            'v_range': v_range,
            'feature_count': len(cluster),
            'missing_pct': round(missing_rate, 2),
            'pca_components': n_comps,
            'explained_variance_ratio': [round(float(v), 4) for v in pca.explained_variance_ratio_]
        }

        summary_rows.append({
            'Cluster': f"Cluster {c_idx}",
            'V_Range': v_range,
            'Feature_Count': len(cluster),
            'Missing_Pct': round(missing_rate, 2),
            'Representative_Feature': best_feat,
            'KS_Statistic': round(ks_dict.get(best_feat, 0.0), 4),
            'PCA_Components': n_comps
        })

    with open(OUTPUTS_DIR / "v_cluster_config.json", "w") as f:
        json.dump(v_cluster_config, f, indent=2)

    with open(OUTPUTS_DIR / "v_pca_dimensions.json", "w") as f:
        json.dump(pca_dimensions, f, indent=2)

    joblib.dump(pca_transformers, OUTPUTS_DIR / "v_pca_transformers.pkl")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS_DIR / "v_clusters_summary.csv", index=False)
    print(f"Saved V-cluster config, PCA models, and {OUTPUTS_DIR / 'v_clusters_summary.csv'}", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    return v_cluster_config, summary_df


def generate_figures(df, ks_df, iv_df, v_cluster_config):
    print("\n[5/5] Generating publication-grade diagnostic figures for paper...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Feature Ranking Plot (KS and IV)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top_ks = ks_df.head(15).sort_values(by='ks_statistic', ascending=True)
    axes[0].barh(top_ks['feature'], top_ks['ks_statistic'], color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].set_title("Top 15 Numerical Features by Kolmogorov-Smirnov (KS) Stat", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("KS Statistic (Separation Power)")
    axes[0].set_xlim(0, max(top_ks['ks_statistic']) * 1.15)
    for i, v in enumerate(top_ks['ks_statistic']):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=9)

    top_iv = iv_df.head(15).sort_values(by='information_value', ascending=True)
    axes[1].barh(top_iv['feature'], top_iv['information_value'], color='#ff7f0e', edgecolor='black', alpha=0.85)
    axes[1].set_title("Top 15 Categorical Features by Information Value (IV)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Information Value (IV)")
    axes[1].set_xlim(0, max(top_iv['information_value']) * 1.15)
    for i, v in enumerate(top_iv['information_value']):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "feature_ranking_plot.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}", flush=True)

    # 2. Concept Drift Plot
    df['Week'] = (df[TIME_COL] // 604800).astype(int)
    weekly_stats = df.groupby('Week').agg(
        total_tx=('isFraud', 'count'),
        fraud_tx=('isFraud', 'sum'),
        fraud_rate=('isFraud', 'mean')
    ).reset_index()

    split_week = TRAIN_VAL_SPLIT_DT / 604800

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    color_bar = '#aec7e8'
    color_line = '#d62728'

    bars = ax1.bar(weekly_stats['Week'], weekly_stats['total_tx'], color=color_bar, alpha=0.7, label='Transaction Volume')
    line = ax2.plot(weekly_stats['Week'], weekly_stats['fraud_rate'] * 100, color=color_line, linewidth=2.5, marker='o', label='Fraud Rate (%)')

    ax1.axvline(x=split_week, color='black', linestyle='--', linewidth=2, label=f'Train/Val Split (~Week {split_week:.1f})')

    ax1.set_xlabel("Chronological Timeline (Weeks from Reference Epoch)", fontweight='bold')
    ax1.set_ylabel("Weekly Transaction Volume", color='#1f77b4', fontweight='bold')
    ax2.set_ylabel("Fraud Rate (%)", color=color_line, fontweight='bold')
    ax1.set_title("Chronological Transaction Volume & Fraud Rate Drift (182-Day IEEE-CIS Benchmark)", fontsize=13, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "concept_drift.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved {fig_path2}", flush=True)

    # 3. V-Cluster Correlation Heatmap
    reps = [v_cluster_config[f"cluster_{i}"]['representative'] for i in range(1, 16)]
    corr_matrix = df[reps].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of 15 V-Cluster Representative Attributes (Multicollinearity Audit)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path3 = FIGURES_DIR / "v_cluster_heatmap.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Saved {fig_path3}", flush=True)


def main():
    print("="*80, flush=True)
    print("DATA MINING CONTRAST & STATISTICAL AUDIT PIPELINE", flush=True)
    print("="*80, flush=True)

    df = load_dataset()

    num_cols = [TRANSACTION_AMT] + C_COLS + D_COLS + DIST_COLS + ID_NUM_COLS + V_COLS
    ks_df = calculate_ks_statistics(df, num_cols, TARGET_COL)

    iv_df = calculate_information_value(df, ALL_CATEGORICAL_FEATURES, TARGET_COL)

    v_cluster_config, summary_df = analyze_v_clusters_and_pca(df, V_COLS, ks_df)

    generate_figures(df, ks_df, iv_df, v_cluster_config)

    print("\n[SUCCESS] Completed data mining audit and generated all diagnostic evidence!", flush=True)


if __name__ == "__main__":
    main()
