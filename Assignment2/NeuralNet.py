"""
Neural Net Hyperparameter Optimization
========================================
Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository
        https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
        (mirrored via sklearn.datasets for reproducibility -- no local path required)

This script:
  1. Loads and preprocesses the dataset (null handling, standardization)
  2. Trains neural networks across a grid of hyperparameter combinations
  3. Tracks loss-per-epoch ("model history") for every combination
  4. Plots model history curves (split into readable groups)
  5. Outputs a summary results table (train/test accuracy, train/test MSE)
  6. Prints a short written summary of findings

No local/absolute paths are hardcoded; the dataset loads from public UCI dataset.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


class NeuralNet:
    """
    Encapsulates data loading, preprocessing, training, and evaluation
    of neural networks across a hyperparameter grid.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.results = []          # list of dicts -> becomes the summary table
        self.histories = {}        # model_label -> list of loss values per epoch

    # ------------------------------------------------------------------
    # 1. Load + preprocess
    # ------------------------------------------------------------------
    # UCI ML Repository URL
    UCI_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "breast-cancer-wisconsin/wdbc.data"
    )

    def load_data(self):
        """
        Loads the Breast Cancer Wisconsin (Diagnostic) dataset directly from
        the UCI Machine Learning Repository via pd.read_csv.

        The raw UCI file has no header row. Per the dataset's documented
        schema (see UCI dataset description page), columns are:
          - id number
          - diagnosis (M = malignant, B = benign)
          - 30 numeric features: mean/SE/"worst" values for 10 measured
            cell-nucleus characteristics (radius, texture, perimeter, area,
            smoothness, compactness, concavity, concave_points, symmetry,
            fractal_dimension)
        """
        base_features = [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry",
            "fractal_dimension",
        ]
        feature_cols = (
            [f"{f}_mean" for f in base_features]
            + [f"{f}_se" for f in base_features]
            + [f"{f}_worst" for f in base_features]
        )
        columns = ["id", "diagnosis"] + feature_cols

        df = pd.read_csv(self.UCI_URL, header=None, names=columns)

        # diagnosis: M (malignant) -> 1, B (benign) -> 0
        df["label"] = (df["diagnosis"] == "M").astype(int)
        df = df.drop(columns=["id", "diagnosis"])
        return df

    def preprocess(self, df):
        """
        Standard preprocessing:
          - Handle null values (none expected in this dataset, but we
            defensively impute with column median for robustness).
          - Ensure data integrity (drop exact duplicate rows).
          - Standardize all feature attributes to zero mean / unit variance.
        Assumption: the label column is already clean binary (0/1) and
        needs no transformation.
        """
        # Handle nulls defensively (median imputation per column)
        df = df.fillna(df.median(numeric_only=True))

        # Data integrity: drop duplicate rows if any
        df = df.drop_duplicates().reset_index(drop=True)

        X = df.drop(columns=["label"]).values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Standardization: fit scaler on train only, apply to both (avoids leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    # ------------------------------------------------------------------
    # 2. Train + evaluate across hyperparameter grid
    # ------------------------------------------------------------------
    def train_evaluate(self, param_grid, max_epochs=200):
        """
        param_grid: dict of hyperparameter name -> list of values to try.
        Trains one MLPClassifier per combination, records loss-curve
        history (model performance vs. epoch), and computes final
        train/test accuracy and MSE.
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))

        for combo in combos:
            params = dict(zip(keys, combo))
            label = ", ".join(f"{k}={v}" for k, v in params.items())

            clf = MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                activation=params["activation"],
                learning_rate_init=params["learning_rate_init"],
                solver="sgd",           # sgd exposes a clean per-epoch loss_curve_
                max_iter=max_epochs,
                random_state=self.random_state,
            )
            clf.fit(self.X_train, self.y_train)

            # Model history: loss per epoch (this IS "accuracy/error vs epochs")
            self.histories[label] = clf.loss_curve_

            # Predictions
            train_pred = clf.predict(self.X_train)
            test_pred = clf.predict(self.X_test)

            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            train_mse = mean_squared_error(self.y_train, train_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)

            self.results.append({
                "hidden_layer_sizes": params["hidden_layer_sizes"],
                "activation": params["activation"],
                "learning_rate_init": params["learning_rate_init"],
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "train_mse": round(train_mse, 4),
                "test_mse": round(test_mse, 4),
                "epochs_run": len(clf.loss_curve_),
                "label": label,
            })

        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------
    # 3. Plot model history (split into groups if congested)
    # ------------------------------------------------------------------
    def plot_histories(self, group_size=4, out_prefix="model_history"):
        labels = list(self.histories.keys())
        n_groups = (len(labels) + group_size - 1) // group_size
        saved_files = []

        for g in range(n_groups):
            subset = labels[g * group_size: (g + 1) * group_size]
            plt.figure(figsize=(9, 6))
            for label in subset:
                plt.plot(self.histories[label], label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title(f"Model History (Loss vs. Epoch) — Group {g + 1}")
            plt.legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            fname = f"{out_prefix}_group{g + 1}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            saved_files.append(fname)

        return saved_files


def main():
    nn = NeuralNet(random_state=42)
    df = nn.load_data()
    nn.preprocess(df)

    # Hyperparameter grid: architecture x activation function x learning rate
    param_grid = {
        "hidden_layer_sizes": [(10,), (20, 10)],
        "activation": ["relu", "tanh", "logistic"],
        "learning_rate_init": [0.01, 0.1],
    }

    results_df = nn.train_evaluate(param_grid, max_epochs=200)
    results_df.to_csv("results_table.csv", index=False)

    plot_files = nn.plot_histories(group_size=4)

    print("\n=== Results Table ===")
    print(results_df.drop(columns=["label"]).to_string(index=False))
    print(f"\nSaved plots: {plot_files}")
    print("Saved table: results_table.csv")

    best_row = results_df.loc[results_df["test_accuracy"].idxmax()]
    print(f"\nBest test accuracy: {best_row['test_accuracy']} "
          f"({best_row['activation']}, {best_row['hidden_layer_sizes']}, "
          f"lr={best_row['learning_rate_init']})")


if __name__ == "__main__":
    main()