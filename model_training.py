"""
QML-IDS  —  Step 3: Training the VQC classifier
Loads preprocessed data, builds the VQC, trains with COBYLA,
and saves the optimal parameters.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from circuit_initialization import build_encoding_circuit, build_ansatz

N_QUBITS    = 4
REPS        = 2
MAX_ITER    = 150
RANDOM_SEED = 42

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

objective_history = []

def callback(weights, obj_func_eval):
    objective_history.append(obj_func_eval)
    if len(objective_history) % 10 == 0:
        print(f"  iter {len(objective_history):>4d}  |  loss = {obj_func_eval:.4f}")


def load_data():
    X_tr = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_te = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_tr = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_te = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"[train] Loaded  X_train{X_tr.shape}  X_test{X_te.shape}")
    return X_tr, X_te, y_tr, y_te


def train(X_tr, y_tr):
    np.random.seed(RANDOM_SEED)

    feature_map  = build_encoding_circuit(N_QUBITS)
    ansatz       = build_ansatz(N_QUBITS, REPS)
    n_params     = ansatz.num_parameters
    init_weights = np.random.uniform(-np.pi, np.pi, n_params)

    optimizer = COBYLA(maxiter=MAX_ITER)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
        initial_point=init_weights,
    )

    print(f"\n[train] Starting COBYLA optimisation  (max_iter={MAX_ITER})")
    print(f"[train] Trainable parameters : {n_params}")
    print(f"[train] Training samples     : {len(y_tr)}\n")

    t0 = time.time()
    vqc.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"\n[train] Training complete in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    return vqc


def plot_loss():
    if not objective_history:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(objective_history, color="#3B8BD4", linewidth=1.5)
    ax.set_xlabel("COBYLA iteration")
    ax.set_ylabel("Objective (cross-entropy loss)")
    ax.set_title("VQC training loss — COBYLA convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "training_loss.png")
    fig.savefig(path, dpi=150)
    print(f"[train] Loss curve saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_data()
    vqc = train(X_tr, y_tr)
    plot_loss()

    train_score = vqc.score(X_tr, y_tr)
    test_score  = vqc.score(X_te, y_te)
    print(f"\n[train] Train accuracy: {train_score*100:.1f}%")
    print(f"[train] Test  accuracy: {test_score*100:.1f}%")

    np.save(os.path.join(MODEL_DIR, "vqc_weights.npy"), vqc.weights)
    print("[train] Trained weights saved → models/vqc_weights.npy")
    print("\n[train] Step 3 complete. Run step4_evaluate.py next.")
