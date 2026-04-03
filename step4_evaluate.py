"""
QML-IDS  —  Step 4: Evaluation & classical baseline comparison
Reconstructs trained VQC from saved weights, computes full metrics,
trains SVM on the same features, and generates all output figures.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, f1_score, accuracy_score
)
from step2_circuit import build_encoding_circuit, build_ansatz

N_QUBITS = 4
REPS     = 2

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    X_tr = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_te = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_tr = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_te = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_tr, X_te, y_tr, y_te


def load_vqc(X_tr, y_tr):
    """
    Reconstruct VQC from saved weights.
    We run a 1-iteration warm-start fit on 4 samples just to initialise
    the object's internal state, then overwrite with the real weights.
    """
    weights = np.load(os.path.join(MODEL_DIR, "vqc_weights.npy"))
    vqc = VQC(
        feature_map=build_encoding_circuit(N_QUBITS),
        ansatz=build_ansatz(N_QUBITS, REPS),
        optimizer=COBYLA(maxiter=1),
        initial_point=weights,
    )
    vqc.fit(X_tr[:4], y_tr[:4])      # warm-start: just primes internal state
    vqc._fit_result.x = weights       # restore real trained weights
    return vqc


def eval_vqc(vqc, X_te, y_te):
    y_pred  = vqc.predict(X_te)
    acc     = accuracy_score(y_te, y_pred)
    f1      = f1_score(y_te, y_pred)
    tpr     = (y_pred[y_te == 1] == 1).sum() / (y_te == 1).sum()
    fpr     = (y_pred[y_te == 0] == 1).sum() / (y_te == 0).sum()
    print(f"[eval] VQC  →  acc={acc*100:.1f}%  F1={f1:.3f}  "
          f"TPR={tpr:.3f}  FPR={fpr:.3f}")
    return y_pred, acc, f1, tpr, fpr


def eval_svm(X_tr, y_tr, X_te, y_te):
    print("[eval] Training SVM baseline (RBF, C=10) …")
    svm = SVC(kernel="rbf", probability=True, random_state=42, C=10)
    svm.fit(X_tr, y_tr)
    y_pred = svm.predict(X_te)
    y_prob = svm.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, y_prob)
    print(f"[eval] SVM  →  acc={acc*100:.1f}%  F1={f1:.3f}  AUC={auc:.3f}")
    return y_pred, y_prob, acc, f1, auc


def plot_confusion_matrices(y_te, y_vqc, y_svm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = [
        (y_vqc, "VQC  (quantum-classical hybrid)"),
        (y_svm, "SVM  (classical baseline)"),
    ]
    for ax, (y_pred, title) in zip(axes, pairs):
        cm = confusion_matrix(y_te, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=11)
    fig.suptitle("QML-IDS: confusion matrices — NSL-KDD test set", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150)
    print(f"[eval] Confusion matrices → {path}")
    plt.close(fig)


def plot_roc(y_te, y_svm_prob, auc_svm, vqc_tpr, vqc_fpr):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr_c, tpr_c, _ = roc_curve(y_te, y_svm_prob)
    ax.plot(fpr_c, tpr_c, color="#3B8BD4", linewidth=2,
            label=f"SVM (classical)  AUC = {auc_svm:.3f}")
    ax.scatter([vqc_fpr], [vqc_tpr], color="#E8593C", s=90, zorder=5,
               label=f"VQC (quantum)  TPR={vqc_tpr:.3f}  FPR={vqc_fpr:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — VQC operating point vs SVM curve\nNSL-KDD test set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    print(f"[eval] ROC curves → {path}")
    plt.close(fig)


def plot_metrics_bar(acc_vqc, f1_vqc, acc_svm, f1_svm, auc_svm):
    metric_labels = ["Accuracy", "F1 score"]
    vqc_vals      = [acc_vqc, f1_vqc]
    svm_vals      = [acc_svm, f1_svm]
    x, w          = np.arange(len(metric_labels)), 0.32

    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(x - w/2, svm_vals, w, label="SVM (classical)",
                color="#3B8BD4", alpha=0.85)
    b2 = ax.bar(x + w/2, vqc_vals, w, label="VQC (quantum)",
                color="#E8593C", alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("QML-IDS: VQC vs SVM — NSL-KDD\n"
                 "4 qubits · 12 parameters · 1500 train / 400 test")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "metrics_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"[eval] Metrics bar chart → {path}")
    plt.close(fig)


def print_reports(y_te, y_vqc, y_svm):
    print("\n── VQC classification report ───────────────────────────────")
    print(classification_report(y_te, y_vqc, target_names=["Normal", "Attack"]))
    print("── SVM classification report ───────────────────────────────")
    print(classification_report(y_te, y_svm, target_names=["Normal", "Attack"]))


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_data()

    vqc = load_vqc(X_tr, y_tr)
    y_vqc, acc_vqc, f1_vqc, vqc_tpr, vqc_fpr = eval_vqc(vqc, X_te, y_te)

    y_svm, y_svm_prob, acc_svm, f1_svm, auc_svm = eval_svm(
        X_tr, y_tr, X_te, y_te)

    print_reports(y_te, y_vqc, y_svm)

    plot_confusion_matrices(y_te, y_vqc, y_svm)
    plot_roc(y_te, y_svm_prob, auc_svm, vqc_tpr, vqc_fpr)
    plot_metrics_bar(acc_vqc, f1_vqc, acc_svm, f1_svm, auc_svm)

    print("\n[eval] All outputs written to outputs/")
    print("[eval] Step 4 complete — project ready.")
