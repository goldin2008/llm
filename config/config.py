import json
import os

import yaml

with open("config/config.json", "r") as f:
    """
    Load config.json and get environment variables
    """
    config = json.load(f)

# with open("config/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# output_folder_path = os.path.join(config['output_folder_path'])
# test_data_path = os.path.join(config['test_data_path'])
# prod_deployment_path = os.path.join(config['prod_deployment_path'])

# INPUT_FOLDER_PATH = os.path.join(os.path.abspath('../'),
#                                  'data',
#                                  config['input_folder_path'])
# DATA_PATH = os.path.join(os.path.abspath('../'),
#                          'data',git
#                          config['output_folder_path'])
# TEST_DATA_PATH = os.path.join(os.path.abspath('../'),
#                               'data',
#                               config['test_data_path'])
# OUTPUT_MODEL_PATH = os.path.join(os.path.abspath('../'),
#                           'model',
#                           config['output_model_path'])
# PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath('../'),
#                                     'model',
#                                     config['prod_deployment_path'])

INPUT_FOLDER_PATH = os.path.join(
    os.path.abspath("./"), config["input_folder_path"]
)
DATA_PATH = os.path.join(os.path.abspath("./"), config["output_folder_path"])
TEST_DATA_PATH = os.path.join(os.path.abspath("./"), config["test_data_path"])
OUTPUT_MODEL_PATH = os.path.join(
    os.path.abspath("./"), config["output_model_path"]
)
PROD_DEPLOYMENT_PATH = os.path.join(
    os.path.abspath("./"), config["prod_deployment_path"]
)
