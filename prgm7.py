import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression_california():
    # Load California Housing Dataset
    housing = fetch_california_housing(as_frame=True)

    X = housing.data[["AveRooms"]]  # Feature: Average number of rooms
    y = housing.target              # Target: Median house value

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Plot Results
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="Actual")
    plt.scatter(X_test, y_pred, color="red", alpha=0.5, label="Predicted")
    plt.xlabel("Average Number of Rooms (AveRooms)")
    plt.ylabel("Median House Value")
    plt.title("Linear Regression - California Housing Dataset")
    plt.legend()
    plt.show()

    # Evaluation
    print("\nLinear Regression - California Housing Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))


def polynomial_regression_auto_mpg():
    # Auto MPG Dataset URL
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "auto-mpg/auto-mpg.data"
    )

    column_names = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]

    # Load Dataset
    data = pd.read_csv(
        url,
        delim_whitespace=True,
        names=column_names,
        na_values="?"
    )

    data = data.dropna()

    # Feature and Target
    X = data[["displacement"]]
    y = data["mpg"]

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Polynomial Regression Model (Degree = 2)
    poly_model = make_pipeline(
        PolynomialFeatures(degree=2),
        StandardScaler(),
        LinearRegression()
    )

    poly_model.fit(X_train, y_train)

    # Predictions
    y_pred = poly_model.predict(X_test)

    # Plot Results
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="Actual")
    plt.scatter(X_test, y_pred, color="red", alpha=0.5, label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("Miles Per Gallon (MPG)")
    plt.title("Polynomial Regression - Auto MPG Dataset")
    plt.legend()
    plt.show()

    # Evaluation
    print("\nPolynomial Regression - Auto MPG Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))


if __name__ == "__main__":
    print("Demonstrating Linear Regression and Polynomial Regression")

    linear_regression_california()
    polynomial_regression_auto_mpg()