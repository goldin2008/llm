import json
import logging
import os
import subprocess

import deployment
import diagnostics
import pandas as pd
import reporting
import scoring
import training
from sklearn import metrics

with open("config/config.json", "r") as config_file:
    config = json.load(config_file)

input_folder_path = os.path.join(config["input_folder_path"])
output_folder_path = os.path.join(config["output_folder_path"])
output_model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

"""
1. Check and read new data
(1) read ingestedfiles.txt
(2) determine whether the source data folder has files that aren't listed in ingestedfiles.txt

2. Deciding whether to proceed, part 1
if you found new data, you should proceed. otherwise, do end the process here

3. Checking for model drift
check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

4. Deciding whether to proceed, part 2
if you found model drift, you should proceed. otherwise, do end the process here

5. Re-deployment
if you found evidence for model drift, re-run the deployment.py script

6. Diagnostics and reporting
run diagnostics.py and reporting.py for the re-deployed model
"""


def check_new_data():
    """
    Checks that the data files are the sames as the one in the ingested file
    :return: bool True if new data is found, False if not
    """
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "r") as f:
        # ingested_files = f.read()
        ingested_files = []
        for line in f:
            line = (
                line.strip()
            )  # Remove leading/trailing whitespace or newline characters
            # Do something with the line
            if line.startswith("Ingestion date"):
                continue
            ingested_files.append(line)  # Or perform any other operations

    source_files = set(os.listdir(input_folder_path))
    diff = source_files.difference(ingested_files)

    print("input_folder_path: ", input_folder_path)
    print("output_folder_path: ", output_folder_path)
    print("ingested_files: ", ingested_files)
    print("source_files: ", source_files)
    print(diff)

    return True if len(diff) > 0 else False


def check_model_drift():
    """
    Checks whether the model has drifted
    :return:
    """
    with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as f:
        latest_score = float(f.read())

    print("latest_score :", latest_score)

    file_path = os.path.join(output_folder_path, "finaldata.csv")
    print("file_path: ", file_path)

    df_data = pd.read_csv(file_path)
    df = df_data.copy().drop("corporation", axis=1)
    y = df["exited"]
    # X = df.drop("exited", axis=1)

    y_pred = diagnostics.model_predictions()
    new_score = metrics.f1_score(y, y_pred)

    return latest_score < new_score


def main():
    """
    Function to run the entire pipeline

    Returns:
    """
    if check_new_data():
        print("Found new data, begin ingestion step")
        subprocess.run(
            ["python", "src/llm/ingestion.py"], stdout=subprocess.PIPE
        )
        if check_model_drift():
            subprocess.run(
                ["python", "src/llm/training.py"], stdout=subprocess.PIPE
            )
            subprocess.run(
                ["python", "src/llm/deployment.py"], stdout=subprocess.PIPE
            )
            subprocess.run(
                ["python", "src/llm/scoring.py"], stdout=subprocess.PIPE
            )
            subprocess.run(
                ["python", "src/llm/diagnostics.py"], stdout=subprocess.PIPE
            )
            subprocess.run(
                ["python", "src/llm/reporting.py"], stdout=subprocess.PIPE
            )


if __name__ == "__main__":
    # check_new_data()
    check_model_drift()
    # main()
