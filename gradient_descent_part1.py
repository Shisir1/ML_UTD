import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

########################################################
# 1. LOAD DATASET
########################################################

# Load the Auto MPG dataset from the UCI Machine Learning Repository
auto_mpg = fetch_ucirepo(id=9)

X = auto_mpg.data.features
y = auto_mpg.data.targets

df = pd.concat([X, y], axis=1)

########################################################
# 2. PREPROCESSING
########################################################

# Remove duplicates
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

# Convert categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# Separate features and target
target_column = 'mpg'

X = df.drop(columns=[target_column])
y = df[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

########################################################
# 3. TRAIN TEST SPLIT
########################################################

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

########################################################
# 4. LINEAR REGRESSION USING GRADIENT DESCENT
########################################################

class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):

        m, n = X.shape

        # Add bias term
        X = np.c_[np.ones(m), X]

        self.weights = np.zeros(n + 1)

        self.cost_history = []

        for _ in range(self.iterations):

            predictions = X.dot(self.weights)

            errors = predictions - y

            gradient = (1 / m) * X.T.dot(errors)

            self.weights -= self.learning_rate * gradient

            cost = (1 / (2 * m)) * np.sum(errors ** 2)

            self.cost_history.append(cost)

    def predict(self, X):

        m = X.shape[0]

        X = np.c_[np.ones(m), X]

        return X.dot(self.weights)

########################################################
# 5. MODEL TUNING
########################################################

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
iterations_list = [1000, 3000, 5000, 10000]

best_mse = float('inf')
best_model = None

log_records = []

for lr in learning_rates:

    for iters in iterations_list:

        model = LinearRegressionGD(
            learning_rate=lr,
            iterations=iters
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_train)

        mse = np.mean((y_train - predictions) ** 2)

        log_records.append({
            "Learning Rate": lr,
            "Iterations": iters,
            "Train MSE": mse
        })

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_lr = lr
            best_iters = iters

########################################################
# SAVE LOG FILE
########################################################

log_df = pd.DataFrame(log_records)

log_df.to_csv(
    "gradient_descent_tuning_log.csv",
    index=False
)

print("\nTuning Results")
print(log_df)

print("\nBest Parameters")
print("Learning Rate:", best_lr)
print("Iterations:", best_iters)
print("Best Training MSE:", best_mse)

########################################################
# 6. TEST DATA EVALUATION
########################################################

test_predictions = best_model.predict(X_test)

test_mse = np.mean(
    (y_test - test_predictions) ** 2
)

rmse = np.sqrt(test_mse)

print("\nTest Results")
print("MSE:", test_mse)
print("RMSE:", rmse)

########################################################
# 7. COST FUNCTION PLOT
########################################################

plt.plot(best_model.cost_history)

plt.title("Cost Function Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost")

plt.show()