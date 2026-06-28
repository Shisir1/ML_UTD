Name: Shisir Humagain

Course: CS 6375 – Machine Learning

Assignment: Assignment 1

---

# Project Overview

This project implements Linear Regression using two approaches:

### Part 1: Custom Gradient Descent Implementation

A multivariate linear regression model was implemented from scratch using Python and NumPy. No machine learning library functions for linear regression or gradient descent were used.

### Part 2: Scikit-Learn SGDRegressor

The same dataset and preprocessing pipeline were used. The model was built using Scikit-Learn's SGDRegressor, which performs linear regression using stochastic gradient descent.

The dataset used is the UCI Auto MPG Dataset.

---

# Dataset

Dataset Name: Auto MPG Dataset

Source: UCI Machine Learning Repository

Target Variable:

* MPG (Miles Per Gallon)

Input Features:

* Cylinders
* Displacement
* Horsepower
* Weight
* Acceleration
* Model Year
* Origin

---

# Libraries Used

## Part 1

* numpy
* pandas
* matplotlib
* sklearn.preprocessing.StandardScaler
* sklearn.model_selection.train_test_split
* ucimlrepo

## Part 2

* numpy
* pandas
* matplotlib
* scikit-learn
* ucimlrepo

Scikit-Learn Components Used:

* SGDRegressor
* StandardScaler
* train_test_split
* mean_squared_error
* mean_absolute_error
* r2_score
* explained_variance_score

---

# Installation

Install required packages:

`pip install numpy pandas matplotlib scikit-learn ucimlrepo`

---

# How to Run Part 1

`python gradient_descent_part1.py`

Output:

* Training results
* Test results
* MSE values
* RMSE values
* Gradient descent convergence graph
* Tuning log file

Generated file:

gradient_descent_tuning_log.csv

---

# How to Run Part 2

`python gradient_descent_part1.py`

Output:

* Best model parameters
* Weight coefficients
* MSE
* RMSE
* MAE
* R² Score
* Explained Variance
* Multiple plots

Generated file:

sgd_regressor_tuning_log.csv

---

# Project Structure

```
Assignment1/
│
├── gradient_descent_part1.py
├── gradient_descent_part2.py
├── README.md
├── gradient_descent_tuning_log.csv
├── sgd_regressor_tuning_log.csv
├── plots/
└── Report.pdf
```
---

# Notes

Feature standardization was applied before training.

Duplicate and missing records were removed during preprocessing.

Categorical variables were converted into numerical values before training.

Multiple learning rates and iteration values were tested to obtain optimal model performance.