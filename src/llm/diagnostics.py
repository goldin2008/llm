import json
import logging
import os
import pickle
import subprocess
import sys
import timeit

import numpy as np
import pandas as pd

from ...config.config import DATA_PATH, PROD_DEPLOYMENT_PATH, TEST_DATA_PATH

# Add the parent_folder directory to the Python import path
parent_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
# print(os.path.dirname(__file__))
# print(parent_folder_path)
sys.path.append(parent_folder_path)

# from config.config import DATA_PATH, PROD_DEPLOYMENT_PATH, TEST_DATA_PATH

with open("config/config.json", "r") as config_file:
    config = json.load(config_file)

input_folder_path = os.path.join(config["input_folder_path"])
output_folder_path = os.path.join(config["output_folder_path"])
output_model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)-15s %(message)s"
)
logger = logging.getLogger()


def load_data():
    """
    Function to get data
    :return:
    """
    logger.info("Load Data")
    file_path = os.path.join(TEST_DATA_PATH, "testdata.csv")
    df_data = pd.read_csv(file_path)
    df = df_data.copy().drop("corporation", axis=1)
    y = df["exited"]
    X = df.drop("exited", axis=1)
    return df_data, X, y


def model_predictions():
    """
    Function to get model predictions
    :return:
    """
    logger.info("Model Predictions")
    _, X, _ = load_data()
    # read the deployed model and a test dataset, calculate predictions
    model_path = os.path.join(PROD_DEPLOYMENT_PATH, "trainedmodel.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(X)
    return y_pred  # return value should be a list containing all predictions


def dataframe_summary():
    """
    Function to get summary statistics
    :return:
    """
    logger.info("Dataframe Summary")
    df, _, _ = load_data()
    # calculate summary statistics here
    # numeric_data = df.drop(['corporation', 'exited'], axis=1)
    # data_summary = numeric_data.agg(['mean', 'median', 'std'])

    data_df = df.drop(["exited"], axis=1)
    data_df = df.select_dtypes("number")
    logging.info("Calculating statistics for data")
    data_summary = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()
        data_summary[col] = {"mean": mean, "median": median, "std": std}

    return data_summary  # return value should be a list containing all summary statistics


def missing_data():
    """
    Function to check missing data
    Calculates percentage of missing data for each column
    in finaldata.csv
    Returns:
        list[dict]: Each dict contains column name and percentage
    """
    logger.info("Loading and preparing finaldata.csv")
    df, _, _ = load_data()
    # data_df = data_df.drop(['corporation', 'exited'], axis=1)

    logging.info("Calculating missing data percentage")
    missing_list = {
        col: {"percentage": perc}
        for col, perc in zip(df.columns, df.isna().sum() / df.shape[0] * 100)
    }

    return missing_list


def _ingestion_timing():
    """
    Function to get timings
    Runs ingestion.py script and measures execution time

    Returns:
        float: running time
    """
    start_time = timeit.default_timer()
    os.system("python src/llm/ingestion.py")
    # os.system(f"python src/llm/ingestion.py")
    # _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - start_time
    return timing


def _training_timing():
    """
    Runs training.py script and measures execution time

    Returns:
        float: running time
    """
    start_time = timeit.default_timer()
    # os.system(f"python src/llm/training.py")
    os.system("python src/llm/training.py")
    # _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - start_time
    return timing


def execution_time():
    """
    Function to get timings

    Returns:
        float: running time
    """
    logger.info("Execution Time")
    # calculate timing of training.py and ingestion.py
    logger.info("Calculating time for ingestion.py")
    ingestion_time = []
    for _ in range(5):
        time = _ingestion_timing()
        ingestion_time.append(time)

    logging.info("Calculating time for training.py")
    training_time = []
    for _ in range(5):
        time = _training_timing()
        training_time.append(time)

    ret_list = [
        {"ingest_time_mean": np.mean(ingestion_time)},
        {"train_time_mean": np.mean(training_time)},
    ]

    return ret_list  # return a list of 2 timing values in seconds


def outdated_packages_list():
    """
    Function to check dependencies
    :return:
    """
    # get a list of
    # outdated = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf-8')

    logger.info("Checking outdated dependencies")
    dependencies = subprocess.run(
        "pip-outdated ./requirements.txt",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        encoding="utf-8",
    )
    # print('dependencies: ', type(dependencies))
    # print('dependencies: ', dependencies)

    # sys.exit()

    dep = dependencies.stdout
    dep = dep.translate(str.maketrans("", "", " \t\r"))
    # dep = dep.split('\n')
    # dep = [dep[3]] + dep[5:-3]
    # dep = [s.split('|')[1:-1] for s in dep]

    return dep


if __name__ == "__main__":
    # y_pred = model_predictions()
    # print(f"y_pred: {y_pred}")

    # data_summary = dataframe_summary()
    # print(f"data_summary: {data_summary}")

    # na_per = missing_data()
    # print(f"na_per: {na_per}")

    # exe_time = execution_time()
    # print(f"exe_time: {exe_time}")

    # outdated = outdated_packages_list()
    # print(f"outdated: {outdated}")

    print(
        "Model predictions on testdata.csv:", model_predictions(), end="\n\n"
    )

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end="\n\n")

    print("Missing percentage")
    print(json.dumps(missing_data(), indent=4), end="\n\n")

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end="\n\n")

    print("Outdated Packages")
    dependencies = outdated_packages_list()
    print(dependencies)
    for row in dependencies:
        print("{:<20}{:<10}{:<10}{:<10}".format(*row))
