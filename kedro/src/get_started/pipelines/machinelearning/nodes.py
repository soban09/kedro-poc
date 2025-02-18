"""
This is a boilerplate pipeline 'machinelearning'
generated using Kedro 0.19.11
"""

import logging
# from typing import dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handle_feature_scaling(data : pd.DataFrame, parameters) -> pd.DataFrame:
    scaler = StandardScaler()
    data[parameters["features"]] = scaler.fit_transform(data[parameters["features"]])
    return data

def handle_min_max_scaling(data : pd.DataFrame, parameters) -> pd.DataFrame:
    scaler = MinMaxScaler()
    data[parameters["features"]] = scaler.fit_transform(data[parameters["features"]])
    return data

def preprocess_data(data : pd.DataFrame, parameters) -> pd.DataFrame:
    data = handle_feature_scaling(data, parameters)
    data = handle_min_max_scaling(data, parameters)
    return data


def split_data(data: pd.DataFrame, parameters):
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters_data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["Diabetic"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        model: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = model.predict(X_test)
    accuracy= round(accuracy_score(y_test, y_pred),2)
    precision = round(precision_score(y_test, y_pred),2)
    recall = round(recall_score(y_test, y_pred),2)
    _f1_score = round(f1_score(y_test, y_pred),2)

    evaluation_metrics = {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1_score" : _f1_score,
    }

    return evaluation_metrics