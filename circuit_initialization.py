"""
QML-IDS  —  Step 2: VQC circuit definition & visualisation
Builds and draws the parameterized ansatz used as the quantum feature extractor.
Run this standalone to inspect the circuit before training.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

N_QUBITS = 4
REPS     = 2          # entanglement repetitions in the ansatz

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def build_encoding_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Angle encoding layer.
    Each qubit i gets RY(x_i) where x_i is the i-th PCA component.
    x is a ParameterVector — Qiskit fills it in at inference time.
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="Encoding")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc


def build_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """
    Variational ansatz: alternating RY rotation layers + CNOT entanglers.
    Uses Qiskit's RealAmplitudes which is the standard choice for
    classification tasks — proven to be expressive at small qubit counts.

    Architecture per rep:
        RY(θ₀)  RY(θ₁)  RY(θ₂)  RY(θ₃)
        CNOT(0→1)  CNOT(1→2)  CNOT(2→3)
    Followed by a final RY layer.
    Total trainable parameters = n_qubits * (reps + 1)
    """
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=reps,
        entanglement="linear",   # CNOT(i → i+1), keeps circuit shallow
        insert_barriers=True,
    )
    return ansatz


def build_full_circuit(n_qubits: int, reps: int) -> QuantumCircuit:
    """
    Compose encoding + ansatz into the full VQC.
    """
    encoding = build_encoding_circuit(n_qubits)
    ansatz   = build_ansatz(n_qubits, reps)

    qc = QuantumCircuit(n_qubits, name="QML-IDS VQC")
    qc.compose(encoding, inplace=True)
    qc.barrier()
    qc.compose(ansatz, inplace=True)
    return qc


def draw_circuit(qc: QuantumCircuit, filename: str):
    fig = qc.draw("mpl", style="clifford", fold=-1)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"[circuit] Saved → {path}")
    plt.close(fig)


def print_circuit_summary(n_qubits, reps):
    ansatz  = build_ansatz(n_qubits, reps)
    n_params = ansatz.num_parameters
    print("\n── VQC architecture summary ────────────────────────────────")
    print(f"  Qubits              : {n_qubits}")
    print(f"  Encoding            : RY(x_i) per qubit  (angle encoding)")
    print(f"  Ansatz              : RealAmplitudes, {reps} reps, linear CNOT")
    print(f"  Trainable params (θ): {n_params}")
    print(f"  Entanglement        : CNOT(0→1), CNOT(1→2), CNOT(2→3)")
    print(f"  Measurement         : Pauli-Z expectation on qubit 0")
    print(f"  Optimizer           : COBYLA (gradient-free)")
    print(f"  Output              : sigmoid(⟨Z₀⟩) → {{normal, attack}}")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    qc = build_full_circuit(N_QUBITS, REPS)

    print_circuit_summary(N_QUBITS, REPS)

    # Full circuit diagram (encoding + ansatz)
    draw_circuit(qc, "vqc_full_circuit.png")

    # Ansatz-only diagram (cleaner for papers/applications)
    ansatz = build_ansatz(N_QUBITS, REPS)
    draw_circuit(ansatz, "vqc_ansatz_only.png")

    print("[circuit] Step 2 complete. Run step3_train.py next.")
