import glob
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Now you can import the module from the subfolder
from config.config import DATA_PATH, OUTPUT_MODEL_PATH, TEST_DATA_PATH

# Add the parent_folder directory to the Python import path
parent_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
# print(os.path.dirname(__file__))
# print(parent_folder_path)
sys.path.append(parent_folder_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("config/config.json", "r") as f:
    """
    Load config.json and get input and output paths
    """
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def merge_multiple_dataframe():
    """
    Function for data ingestion
    :return:
    """
    # check for datasets, compile them together, and write to an output file
    df = pd.DataFrame()

    # recursivly search direcotries and read .csv files.
    datasets = glob.glob(f"{input_folder_path}/*.csv", recursive=True)
    df = pd.concat(map(pd.read_csv, datasets))

    df_final = df.drop_duplicates()
    df_final.to_csv(
        os.path.join(output_folder_path, "finaldata.csv"), index=False
    )

    file_list = [os.path.basename(filepath) for filepath in datasets]

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        f.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        )
        for file in file_list:
            f.write(file + "\n")


if __name__ == "__main__":
    merge_multiple_dataframe()
