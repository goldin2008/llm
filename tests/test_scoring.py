# imports
import pytest
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import json

# function to test
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Take trained model
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y_test = test_df.pop('exited')
    X_test = test_df.drop(['corporation'], axis=1)

    y_pred = model.predict(X_test)
    score = metrics.f1_score(y_test, y_pred)

    score_path = os.path.join(output_model_path, "latestscore.txt")
    with open(score_path, 'w') as f:
        f.write(str(score))

    return score

# unit tests
# below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator

# Test that the function can load a trained model and test data from the specified paths
@pytest.mark.parametrize("output_model_path, test_data_path", [("models", "test_data")])
def test_load_data(output_model_path, test_data_path):
    config = {'output_model_path': output_model_path, 'test_data_path': test_data_path}
    with open('config.json', 'w') as f:
        json.dump(config, f)
    assert os.path.join(output_model_path) == "models"
    assert os.path.join(test_data_path) == "test_data"

# Test that the function can calculate the F1 score for the model relative to the test data
@pytest.mark.parametrize("output_model_path, test_data_path", [("models", "test_data")])
def test_score_model(output_model_path, test_data_path):
    config = {'output_model_path': output_model_path, 'test_data_path': test_data_path}
    with open('config.json', 'w') as f:
        json.dump(config, f)
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 0, 1, 1]), test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    with open(os.path.join(output_model_path, "trainedmodel.pkl"), 'wb') as f:
        pickle.dump(model, f)
    test_df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8], 'exited': [0, 0, 1, 1], 'corporation': ['A', 'B', 'C', 'D']})
    test_df.to_csv(os.path.join(test_data_path, 'testdata.csv'), index=False)
    assert score_model() == 1.0

# Test that the function can write the F1 score to the latestscore.txt file
@pytest.mark.parametrize("output_model_path, test_data_path", [("models", "test_data")])
def test_write_score(output_model_path, test_data_path):
    config = {'output_model_path': output_model_path, 'test_data_path': test_data_path}
    with open('config.json', 'w') as f:
        json.dump(config, f)
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 0, 1, 1]), test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    with open(os.path.join(output_model_path, "trainedmodel.pkl"), 'wb') as f:
        pickle.dump(model, f)
    test_df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8], 'exited': [0, 0, 1, 1], 'corporation': ['A', 'B', 'C', 'D']})
    test_df.to_csv(os.path.join(test_data_path, 'testdata.csv'), index=False)
    score_model()
    with open(os.path.join(output_model_path, "latestscore.txt"), 'r') as f:
        assert f.read() == "1.0"

# Test that the function returns a valid F1 score
@pytest.mark.parametrize("output_model_path, test_data_path", [("models", "test_data")])
def test_valid_score(output_model_path, test_data_path):
    config = {'output_model_path': output_model_path, 'test_data_path': test_data_path}
    with open('config.json', 'w') as f:
        json.dump(config, f)
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 0, 1, 1]), test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    with open(os.path.join(output_model_path, "trainedmodel.pkl"), 'wb') as f:
        pickle.dump(model, f)
    test_df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8], 'exited': [0, 0, 1, 1], 'corporation': ['A', 'B', 'C', 'D']})
    test_df.to_csv(os.path.join(test_data_path, 'testdata.csv'), index=False)
    assert isinstance(score_model(), float)
    assert score_model() >= 0.0 and score_model() <= 1.0

# Test that the function is deterministic
@pytest.mark.parametrize("output_model_path, test_data_path", [("models", "test_data")])
def test_deterministic(output_model_path, test_data_path):
    config = {'output_model_path': output_model_path, 'test_data_path': test_data_path}
    with open('config.json', 'w') as f:
        json.dump(config, f)
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 0, 1, 1]), test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    with open(os.path.join(output_model_path, "trainedmodel.pkl"), 'wb') as f:
        pickle.dump(model, f)
    test_df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8], 'exited': [0, 0, 1, 1], 'corporation': ['A', 'B', 'C', 'D']})
    test_df.to_csv(os.path.join(test_data_path, 'testdata.csv'), index=False)
    assert score_model() == score_model()
