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

output_model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])


def score_model():
    """
    Function for model scoring
    :return:
    """
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    # Take trained model
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    y_test = test_df.pop("exited")
    X_test = test_df.drop(["corporation"], axis=1)

    y_pred = model.predict(X_test)
    score = metrics.f1_score(y_test, y_pred)

    score_path = os.path.join(output_model_path, "latestscore.txt")
    with open(score_path, "w") as f:
        f.write(str(score))

    return score


if __name__ == "__main__":
    score = score_model()
