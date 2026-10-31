from pathlib import Path

# =============================================================================
# DIRECTORY & FILE PATH CONFIGURATION (Single Source of Truth)
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw" / "ieee-fraud-detection"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

# Ensure all primary working directories exist
for directory in [DATA_PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, SAVED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_TRANS_PATH = DATA_RAW_DIR / "train_transaction.csv"
TRAIN_IDENT_PATH = DATA_RAW_DIR / "train_identity.csv"
TEST_TRANS_PATH = DATA_RAW_DIR / "test_transaction.csv"
TEST_IDENT_PATH = DATA_RAW_DIR / "test_identity.csv"
SAMPLE_SUB_PATH = DATA_RAW_DIR / "sample_submission.csv"

# =============================================================================
# OFFICIAL 49 CATEGORICAL ATTRIBUTES SPECIFICATION
# (31 string/object columns + 18 numeric columns treated as categorical entities)
# =============================================================================
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

ALL_CATEGORICAL_FEATURES = OFFICIAL_CAT_TRANS + OFFICIAL_CAT_IDENT

# =============================================================================
# NUMERICAL AND ANONYMIZED FEATURE SPECIFICATION
# =============================================================================
TRANSACTION_AMT = 'TransactionAmt'
C_COLS = [f'C{i}' for i in range(1, 15)]
D_COLS = [f'D{i}' for i in range(1, 16)]
DIST_COLS = ['dist1', 'dist2']
ID_NUM_COLS = [f'id_{i:02d}' for i in range(1, 12)]
V_COLS = [f'V{i}' for i in range(1, 340)]

BASE_NUMERICAL_FEATURES = [TRANSACTION_AMT] + C_COLS + D_COLS + DIST_COLS + ID_NUM_COLS

TARGET_COL = 'isFraud'
ID_COL = 'TransactionID'
TIME_COL = 'TransactionDT'

# Temporal split threshold: 75% of train chronologically (no future leakage)
TRAIN_VAL_SPLIT_DT = 11246665

# Entity Embedding dimension calculation rule: d_i = min(50, int(K_i**0.5 / 2) + 2)
def calc_embedding_dim(cardinality: int) -> int:
    return min(50, int(cardinality**0.5 / 2) + 2)

# Training Hyperparameters
RANDOM_SEED = 42
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
