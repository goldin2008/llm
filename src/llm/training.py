import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, session
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

with open("config/config.json", "r") as f:
    """
    Load config.json and get path variables
    """
    config = json.load(f)

output_folder_path = os.path.join(config["output_folder_path"])
output_model_path = os.path.join(config["output_model_path"])


def train_model():
    """
    Function for training the model

    Args:
    Returns:

    The result of the addition process
    """
    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    # encode corporation code to numeric value
    df["corporation"] = df["corporation"].apply(
        lambda x: sum(bytearray(x, "utf-8"))
    )
    df = df.drop("corporation", axis=1)
    y = df["exited"]
    X = df.drop("exited", axis=1)
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(config["output_model_path"]):
        os.makedirs(config["output_model_path"])

    model_output_path = os.path.join(
        config["output_model_path"], "trainedmodel.pkl"
    )

    pickle.dump(model, open(model_output_path, "wb"))


if __name__ == "__main__":
    train_model()
