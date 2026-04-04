# QML-IDS: Quantum-Classical Hybrid Anomaly Detection for Network Intrusion Detection

A proof-of-concept implementation of a **Variational Quantum Circuit (VQC)**-based
intrusion detection system, benchmarked against a classical SVM on the NSL-KDD dataset.

Built as part of a research portfolio exploring quantum machine learning applications
in cybersecurity. Runs entirely on the local **Qiskit Aer simulator** — no cloud
account or IBM Quantum access required.

---

## Architecture

```
NSL-KDD features
      │
      ▼
Classical preprocessing
  StandardScaler + PCA (15 → 4 dimensions)
      │
      ▼
Angle encoding  ──  RY(x_i) on qubit i
      │
      ▼
Variational Quantum Circuit (VQC)
  RealAmplitudes ansatz  |  4 qubits  |  2 reps
  CNOT entanglement (linear)
  12 trainable parameters θ
      │
      ▼
Pauli-Z measurement  →  ⟨Z₀⟩ ∈ [-1, 1]
      │
      ▼
sigmoid  →  binary classification
  {0: normal,  1: attack}
      │
      ▼
COBYLA optimiser  (gradient-free, 150 iterations)
```

---

## Why a hybrid quantum-classical approach for IDS?

Classical IDS models operate in the original feature space. A VQC implicitly maps
features into a **2ⁿ-dimensional Hilbert space** via entanglement — for 4 qubits
that is a 16-dimensional space from 4 input features. The hypothesis is that
attack and normal traffic patterns may be more **linearly separable** in this
richer representation than in the original space.

This is consistent with the quantum kernel interpretation of VQCs
(Havlíček et al., *Nature* 2019): the quantum circuit computes an inner product
in Hilbert space that is classically hard to simulate at scale.

At 4–6 qubits on a noiseless simulator, we do not expect to outperform a
well-tuned SVM — the value of this work is validating the **pipeline architecture**
and encoding approach for near-term quantum hardware.

---

## Dataset

**NSL-KDD** (improved version of KDD Cup 99)
- 15 numeric features selected from 41 total
- Binary classification: `normal` vs `attack`
- 1,500 training samples / 400 test samples (stratified)
- Subset used to keep simulation time tractable

---

## Requirements

```
python >= 3.10
qiskit == 1.3.2
qiskit-aer == 0.15.1
qiskit-machine-learning == 0.8.2
scikit-learn == 1.4.2
matplotlib == 3.9.0
pandas == 2.2.2
numpy == 1.26.4
```

Install:
```bash
pip install qiskit==1.3.2 qiskit-aer==0.15.1 qiskit-machine-learning==0.8.2 \
            scikit-learn==1.4.2 matplotlib pandas numpy pylatexenc
```

---

## Running the project

```bash
# Step 1 — Download NSL-KDD, preprocess, PCA, angle-encode, save splits
python data_process.py

# Step 2 — Build and visualise the VQC circuit (no training)
python circuit_initialization.py

# Step 3 — Train the VQC classifier (~10–20 min on CPU)
python model_training.py

# Step 4 — Evaluate VQC vs SVM, generate all figures
python evaluate_model.py
```

All outputs (figures, trained weights) are written to `outputs/`.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/pca_scatter.png` | NSL-KDD in PCA space after angle encoding |
| `outputs/vqc_full_circuit.png` | Full circuit diagram (encoding + ansatz) |
| `outputs/vqc_ansatz_only.png` | Ansatz diagram (for papers/slides) |
| `outputs/training_loss.png` | COBYLA convergence curve |
| `outputs/confusion_matrices.png` | VQC vs SVM confusion matrices |
| `outputs/roc_curves.png` | ROC curve comparison |
| `outputs/metrics_comparison.png` | Accuracy / F1 / AUC bar chart |

---

## Limitations & future work

- **Qubit count**: 4–6 qubits insufficient for practical quantum advantage; this is a near-term NISQ demonstration.
- **Noise**: Aer simulator is noiseless; real hardware would degrade performance significantly.
- **Scalability**: COBYLA scales poorly beyond ~50 parameters; SPSA or gradient-based optimisers (parameter shift rule) needed for deeper circuits.
- **Feature set**: Only 4 of 41 KDD features retained after PCA; feature selection strategies could improve the quantum encoding.

**Planned extensions**: noise model injection via Aer's `NoiseModel`, qubit count scaling study, SPSA optimiser comparison, multi-class attack classification.

---

## References

1. Havlíček et al. — *Supervised learning with quantum-enhanced feature spaces*, Nature 2019
2. Cerezo et al. — *Variational quantum algorithms*, Nature Reviews Physics 2021
3. Tavallaee et al. — *A detailed analysis of the KDD CUP 99 dataset*, IEEE 2009
4. Qiskit Machine Learning documentation — https://qiskit-community.github.io/qiskit-machine-learning/
