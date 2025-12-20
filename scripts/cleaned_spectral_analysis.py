"""
Spectral Feature Selection via MSE Minimization
----------------------------------------------
This script is a cleaned and structured version of the original
exploratory notebook. It processes spectral patient data, performs
column-wise regression-based evaluation, and identifies the most
informative wavenumbers based on MSE statistics.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------
# Configuration
# -------------------------
DATA_PATH = "patients.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------
# Load and preprocess data
# -------------------------
df = pd.read_csv(DATA_PATH)

# Clean column naming precision
df["798.385451"] = df["798.385451"].round(3)
df = df.rename(columns={"798.385451": "798.385"})

# Transpose to make samples rows
df = df.T.reset_index()
df.columns = df.iloc[0]
df = df.iloc[1:]

# Convert to numeric
df = df.apply(pd.to_numeric)

# -------------------------
# Feature-target split
# -------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -------------------------
# Column-wise evaluation
# -------------------------
mse_all_columns = []

for col in tqdm(X.columns):
    X_col = X[[col]]
    mse_runs = []

    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X_col, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_runs.append(mean_squared_error(y_test, y_pred))

    mse_all_columns.append(np.mean(mse_runs))

# -------------------------
# Best feature identification
# -------------------------
best_index = int(np.argmin(mse_all_columns))
best_wavenumber = X.columns[best_index]
best_mse = mse_all_columns[best_index]

print(f"Best wavenumber: {best_wavenumber}")
print(f"Minimum MSE: {best_mse}")
