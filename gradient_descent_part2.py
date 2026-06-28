import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

########################################################
# DATA PROCESSOR CLASS
########################################################

class DataProcessor:

    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self):

        auto_mpg = fetch_ucirepo(id=9)

        X = auto_mpg.data.features
        y = auto_mpg.data.targets

        df = pd.concat([X, y], axis=1)

        return df

    def preprocess(self, df):

        df = df.drop_duplicates()

        df = df.dropna()

        categorical_cols = df.select_dtypes(
            include=['object']
        ).columns

        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]

        X = df.drop(columns=['mpg'])

        y = df['mpg']

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y


########################################################
# SGD REGRESSION CLASS
########################################################

class SGDRegressionModel:

    def __init__(
            self,
            learning_rate='constant',
            eta0=0.01,
            max_iter=1000,
            alpha=0.0001):

        self.model = SGDRegressor(
            loss='squared_error',
            learning_rate=learning_rate,
            eta0=eta0,
            max_iter=max_iter,
            alpha=alpha,
            random_state=42
        )

    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)

    def predict(self, X):

        return self.model.predict(X)

    def coefficients(self):

        return self.model.coef_

    def intercept(self):

        return self.model.intercept_


########################################################
# EVALUATION CLASS
########################################################

class Evaluator:

    @staticmethod
    def calculate_metrics(y_true, y_pred):

        mse = mean_squared_error(y_true, y_pred)

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_true, y_pred)

        r2 = r2_score(y_true, y_pred)

        explained_var = explained_variance_score(
            y_true,
            y_pred
        )

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Explained Variance": explained_var
        }


########################################################
# MAIN PROGRAM
########################################################

processor = DataProcessor()

df = processor.load_data()

X, y = processor.preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

########################################################
# HYPERPARAMETER TUNING
########################################################

eta_values = [0.0001, 0.001, 0.01, 0.05]

iteration_values = [1000, 3000, 5000]

alpha_values = [0.0001, 0.001]

best_r2 = -999

best_model = None

log_records = []

for eta in eta_values:

    for max_iter in iteration_values:

        for alpha in alpha_values:

            model = SGDRegressionModel(
                eta0=eta,
                max_iter=max_iter,
                alpha=alpha
            )

            model.train(
                X_train,
                y_train
            )

            train_pred = model.predict(X_train)

            test_pred = model.predict(X_test)

            train_mse = mean_squared_error(
                y_train,
                train_pred
            )

            test_mse = mean_squared_error(
                y_test,
                test_pred
            )

            r2 = r2_score(
                y_test,
                test_pred
            )

            log_records.append({
                "Learning Rate": eta,
                "Iterations": max_iter,
                "Alpha": alpha,
                "Train MSE": train_mse,
                "Test MSE": test_mse,
                "Test R2": r2
            })

            if r2 > best_r2:
                best_r2 = r2
                best_model = model

########################################################
# SAVE TUNING LOG
########################################################

log_df = pd.DataFrame(log_records)

log_df.to_csv(
    "sgd_regressor_tuning_log.csv",
    index=False
)

########################################################
# FINAL EVALUATION
########################################################

predictions = best_model.predict(X_test)

metrics = Evaluator.calculate_metrics(
    y_test,
    predictions
)

print("\nBEST MODEL RESULTS")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

########################################################
# COEFFICIENTS
########################################################

print("\nFeature Coefficients")

for idx, coef in enumerate(
        best_model.coefficients()):
    print(f"Feature {idx}: {coef}")

print(
    "\nIntercept:",
    best_model.intercept()
)

########################################################
# PLOT 1
# ACTUAL VS PREDICTED
########################################################

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual MPG")

plt.ylabel("Predicted MPG")

plt.title(
    "Actual vs Predicted MPG"
)

plt.show()

########################################################
# PLOT 2
# RESIDUALS
########################################################

residuals = y_test - predictions

plt.figure(figsize=(8,6))

plt.scatter(
    predictions,
    residuals
)

plt.axhline(0)

plt.xlabel("Predicted MPG")

plt.ylabel("Residuals")

plt.title(
    "Residual Plot"
)

plt.show()

########################################################
# PLOT 3
# MSE VS ITERATIONS
########################################################

summary = log_df.groupby(
    "Iterations"
)["Test MSE"].mean()

plt.figure(figsize=(8,6))

plt.plot(
    summary.index,
    summary.values,
    marker='o'
)

plt.xlabel("Iterations")

plt.ylabel("Average Test MSE")

plt.title(
    "MSE vs Iterations"
)

plt.show()