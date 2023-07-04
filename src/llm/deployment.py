import json
import os
import pickle
from shutil import copy2

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, session
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

with open("config/config.json", "r") as f:
    """
    Load config.json
    """
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])


def deploy_model():
    """
    Function for deployment
    :return:
    """
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    file_dict = {
        "ingestedfiles.txt": dataset_csv_path,
        "trainedmodel.pkl": output_model_path,
        "latestscore.txt": output_model_path,
    }

    if not os.path.exists(config["prod_deployment_path"]):
        os.makedirs(config["prod_deployment_path"])

    for file, path in file_dict.items():
        copy2(os.path.join(path, file), prod_deployment_path)


if __name__ == "__main__":
    deploy_model()
