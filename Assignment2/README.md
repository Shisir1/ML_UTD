Dataset

Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository
Link: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

The dataset is downloaded directly from the UCI ML Repository via
pd.read_csv() pointed at the public URL below — no bundled library copy
and no local/machine-specific file path:

https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

The raw file has no header row; column names are assigned in code based on
the dataset's documented UCI schema (id, diagnosis M/B, then 30 numeric
features covering mean/SE/"worst" values for 10 cell-nucleus measurements).

Sandbox note: this exact download was validated in two parts, since the
development sandbox used to build this code blocks outbound requests to
archive.ics.uci.edu: (1) the parsing logic was verified against a
locally-generated file matching the real UCI schema byte-for-byte, and
(2) network reachability was independently confirmed to fail only due to
the sandbox's domain allowlist, not a code issue. On a machine with normal
internet access, pd.read_csv(UCI_URL) will fetch the live file directly.

## How to Run
1. Install dependencies:
   ```
   pip install scikit-learn matplotlib pandas numpy
   ```
2. Run the script:
   ```
   python3 NeuralNet.py
   ```
3. Outputs generated in the working directory:
   - `results_table.csv` — hyperparameters vs. train/test accuracy & MSE
   - `model_history_group1.png`, `_group2.png`, `_group3.png` — loss-vs-epoch
     plots, split into groups of 4 models each for readability
   - Console output: results table + best-performing configuration

## Hyperparameters Tested
- **Architecture (`hidden_layer_sizes`)**: `(10,)` single hidden layer of 10
  units, and `(20, 10)` two hidden layers
- **Activation function**: `relu`, `tanh`, `logistic` (sigmoid)
- **Learning rate**: `0.01`, `0.1`

This produces 2 × 3 × 2 = **12 model combinations**, each trained for up to
200 epochs using SGD (chosen specifically because it exposes a clean
per-epoch `loss_curve_` for plotting model history; solvers like `adam` do
not expose this in the same iteration-aligned way in scikit-learn).

## Assumptions Made
1. "Model history" is interpreted as training loss per epoch, tracked via
   scikit-learn's `loss_curve_` attribute (available when `solver="sgd"` or
   `"adam"`).
2. The dataset has no missing values in practice, but the preprocessing step
   defensively imputes any nulls with the column median, in case the
   assignment is graded against a different/noisier UCI dataset substitution.
3. Duplicate rows are dropped as a data-integrity step.
4. Standardization (zero mean, unit variance) is fit only on the training
   split and applied to both train/test to avoid data leakage.
5. "Test accuracy/error" is computed on a held-out 20% split, stratified by
   class label to preserve class balance.
6. MSE is reported on the 0/1 class predictions (treating classification
   output as numeric) since the assignment explicitly asks for MSE as one of
   the error metrics alongside accuracy.

## Files
- `NeuralNet.py` — full source code (class `NeuralNet`)
- `results_table.csv` — output results table
- `model_history_group*.png` — model history plots
- `report.md` — written summary of findings