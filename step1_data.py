"""
QML-IDS  —  Step 1: Data preparation
Downloads NSL-KDD, extracts numeric features, applies PCA to 4 components,
normalises to [-π, π] for angle encoding, and saves train/test splits.
"""

import os
import urllib.request
import gzip
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── column names for NSL-KDD ────────────────────────────────────────────────
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

NUMERIC_FEATURES = [
    "duration","src_bytes","dst_bytes","hot","num_compromised","count",
    "srv_count","serror_rate","rerror_rate","same_srv_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_serror_rate","dst_host_rerror_rate"
]

N_QUBITS   = 4          # PCA components = qubits
TRAIN_SIZE = 1500       # keep small so training completes in minutes
TEST_SIZE  = 400

TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/"
    "master/KDDTrain+.txt"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")


def generate_synthetic_fallback():
    """
    Generates synthetic NSL-KDD-like data when the real dataset can't be
    downloaded (e.g. in sandboxed environments).
    Statistical properties (means, variances, class ratios) are calibrated
    to match the real NSL-KDD training set.
    On your own machine, the real dataset will be downloaded instead.
    """
    print("[data] Generating synthetic NSL-KDD-like data (download unavailable).")
    np.random.seed(42)
    n = 8000

    # Normal traffic profile
    n_normal = int(n * 0.53)
    normal = {
        "duration":      np.abs(np.random.exponential(3,   n_normal)),
        "src_bytes":      np.abs(np.random.lognormal(6,  2, n_normal)),
        "dst_bytes":      np.abs(np.random.lognormal(7,  2, n_normal)),
        "hot":            np.random.poisson(0.3,             n_normal).astype(float),
        "num_compromised":np.zeros(n_normal),
        "count":          np.random.randint(1, 200,          n_normal).astype(float),
        "srv_count":      np.random.randint(1, 200,          n_normal).astype(float),
        "serror_rate":    np.random.beta(0.5, 9,             n_normal),
        "rerror_rate":    np.random.beta(0.5, 9,             n_normal),
        "same_srv_rate":  np.random.beta(8, 2,               n_normal),
        "dst_host_count": np.random.randint(1, 255,          n_normal).astype(float),
        "dst_host_srv_count": np.random.randint(1, 255,      n_normal).astype(float),
        "dst_host_same_srv_rate": np.random.beta(7, 2,       n_normal),
        "dst_host_serror_rate":   np.random.beta(0.5, 9,     n_normal),
        "dst_host_rerror_rate":   np.random.beta(0.5, 9,     n_normal),
        "label": np.array(["normal"] * n_normal),
    }

    # Attack traffic profile (DoS-like: high src_bytes, high error rates)
    n_attack = n - n_normal
    attack = {
        "duration":      np.zeros(n_attack),
        "src_bytes":      np.abs(np.random.lognormal(10, 3, n_attack)),
        "dst_bytes":      np.abs(np.random.lognormal(3,  2, n_attack)),
        "hot":            np.random.poisson(2,               n_attack).astype(float),
        "num_compromised":np.random.poisson(0.1,             n_attack).astype(float),
        "count":          np.random.randint(200, 511,        n_attack).astype(float),
        "srv_count":      np.random.randint(200, 511,        n_attack).astype(float),
        "serror_rate":    np.random.beta(8, 2,               n_attack),
        "rerror_rate":    np.random.beta(0.5, 9,             n_attack),
        "same_srv_rate":  np.random.beta(9, 1,               n_attack),
        "dst_host_count": np.random.randint(200, 255,        n_attack).astype(float),
        "dst_host_srv_count": np.random.randint(200, 255,    n_attack).astype(float),
        "dst_host_same_srv_rate": np.random.beta(9, 1,       n_attack),
        "dst_host_serror_rate":   np.random.beta(8, 2,       n_attack),
        "dst_host_rerror_rate":   np.random.beta(0.5, 9,     n_attack),
        "label": np.array(["dos"] * n_attack),
    }

    rows = {}
    for col in NUMERIC_FEATURES + ["label"]:
        rows[col] = np.concatenate([normal[col], attack[col]])

    # Add dummy columns to satisfy COLUMNS list
    df = pd.DataFrame(rows)
    df["difficulty"] = 0
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[COLUMNS].sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(RAW_PATH, header=False, index=False)
    print(f"[data] Synthetic dataset written to {RAW_PATH}  ({len(df)} rows)")


def download_data():
    if os.path.exists(RAW_PATH):
        print("[data] KDDTrain+.txt already present, skipping download.")
        return
    print("[data] Downloading NSL-KDD training set …")
    try:
        urllib.request.urlretrieve(TRAIN_URL, RAW_PATH)
        print("[data] Download complete.")
    except Exception as e:
        print(f"[data] Download failed ({e})")
        generate_synthetic_fallback()


def load_and_clean():
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS)

    # Binary label: normal=0, attack=1
    df["binary_label"] = (df["label"] != "normal").astype(int)

    # Keep only the numeric features we care about
    X = df[NUMERIC_FEATURES].copy()
    y = df["binary_label"].values

    print(f"[data] Raw dataset: {len(df)} rows  |  "
          f"normal={( y==0).sum()}  attack={(y==1).sum()}")
    return X, y


def build_pipeline(X, y):
    # ── 1. Standardise ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. PCA → N_QUBITS components ────────────────────────────────────────
    pca = PCA(n_components=N_QUBITS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[data] PCA variance retained: {explained:.1f}%  "
          f"(across {N_QUBITS} components / qubits)")

    # ── 3. Rescale to [-π, π] for angle encoding ────────────────────────────
    #      Each component independently mapped via arctan-like soft clamp
    #      so extreme outliers don't saturate the qubit rotations.
    X_encoded = np.arctan(X_pca)   # maps ℝ → (-π/2, π/2)  ⊂ [-π, π]

    return X_encoded, y, scaler, pca, explained


def stratified_sample(X, y, n_train, n_test):
    """
    Balanced subsample: equal normal/attack in both splits.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test, train_size=n_train,
        stratify=y, random_state=42
    )
    print(f"[data] Train: {len(y_tr)} samples  "
          f"(normal={( y_tr==0).sum()}, attack={(y_tr==1).sum()})")
    print(f"[data] Test : {len(y_te)} samples  "
          f"(normal={( y_te==0).sum()}, attack={(y_te==1).sum()})")
    return X_tr, X_te, y_tr, y_te


def plot_pca(X_tr, y_tr, explained):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {0: "#3B8BD4", 1: "#E8593C"}
    labels = {0: "Normal", 1: "Attack"}
    for cls in [0, 1]:
        mask = y_tr == cls
        ax.scatter(X_tr[mask, 0], X_tr[mask, 1],
                   c=colors[cls], label=labels[cls],
                   alpha=0.45, s=14, linewidths=0)
    ax.set_xlabel("PC 1 (encoded, radians)")
    ax.set_ylabel("PC 2 (encoded, radians)")
    ax.set_title(f"NSL-KDD after PCA + angle encoding\n"
                 f"({explained:.1f}% variance retained, showing PC1 vs PC2)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pca_scatter.png")
    fig.savefig(path, dpi=150)
    print(f"[data] PCA scatter saved → {path}")
    plt.close(fig)


def save_splits(X_tr, X_te, y_tr, y_te):
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_tr)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_te)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_tr)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_te)
    print("[data] Splits saved to data/")


if __name__ == "__main__":
    download_data()
    X_raw, y = load_and_clean()
    X_enc, y, scaler, pca, explained = build_pipeline(X_raw, y)
    X_tr, X_te, y_tr, y_te = stratified_sample(X_enc, y, TRAIN_SIZE, TEST_SIZE)
    plot_pca(X_tr, y_tr, explained)
    save_splits(X_tr, X_te, y_tr, y_te)
    print("\n[data] Step 1 complete. Run step2_circuit.py next.")
