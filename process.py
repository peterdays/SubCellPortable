import argparse
import datetime
import logging
import os
import sys

import requests
from skimage.io import imread, imsave
import yaml

import inference


# This is the log configuration. It will log everything to a file AND the console
logging.basicConfig(filename='log.txt', encoding='utf-8', format='%(levelname)s: %(message)s', filemode='w', level=logging.INFO)
console = logging.StreamHandler()
logging.getLogger().addHandler(console)
logger = logging.getLogger("SubCell inference")

# This is the general configuration variable. We are going to use the special key "log" in the dictionary to use the log in our code
config = { "log": logger}

# If you want to use constants with your script, add them here
config["model_channels"] = "rybg"
config["model_type"] = "mae_contrast_supcon_model"
config["update_model"] = False

# If you want to use command line parameters with your script, add them here
if len(sys.argv) > 1:
    argparser = argparse.ArgumentParser(description="Please input the following parameters")
    argparser.add_argument("-c", "--model_channels", help="channel images to be used [rybg, rbg, ybg, bg]", default="rybg", type=str)
    argparser.add_argument("-t", "--model_type", help="model type to be used [mae_contrast_supcon_model, vit_supcon_model]", default="mae_contrast_supcon_model", type=str)
    argparser.add_argument("-u", "--update_model", help="if you want to update the selected model files [True, False]", default=False, type=bool)
    args = argparser.parse_args()
    config = config | args.__dict__

# If you want to use a configuration file with your script, add it here
with open("config.yaml", "r") as file:
    config_contents = yaml.safe_load(file)
    if config_contents:
        config = config | config_contents

# Log the start time and the final configuration so you can keep track of what you did
config["log"].info('Start: ' + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
config["log"].info('Parameters used:')
config["log"].info(config)
config["log"].info('----------')


try:
    classifier_path = os.path.join("models", config["model_channels"], config["model_type"], "classifier.pth")
    encoder_path = os.path.join("models", config["model_channels"], config["model_type"], "encoder.pth")
    # Checking for model update
    if not os.path.isfile(classifier_path) or not os.path.isfile(encoder_path) or config["update_model"]:
        config["log"].info("- Downloading models...")
        with open('models_urls.yaml', 'r') as urls_file:
            url_info = yaml.safe_load(urls_file)

            response = requests.get(url_info[config["model_channels"]][config["model_type"]]["classifier"])
            if response.status_code == 200:
                with open(os.path.join("models", config["model_channels"], config["model_type"], "classifier.pth"), 'wb') as downloaded_file:
                    downloaded_file.write(response.content)
                config["log"].info("  - classifier.pth updated.")
            else:
                config["log"].warning("  - classifier.pth url not found.")

            response = requests.get(url_info[config["model_channels"]][config["model_type"]]["encoder"])
            if response.status_code == 200:
                with open(os.path.join("models", config["model_channels"], config["model_type"], "encoder.pth"), 'wb') as downloaded_file:
                    downloaded_file.write(response.content)
                config["log"].info("  - encoder.pth updated.")
            else:
                config["log"].warning("  - encoder.pth url not found.")

    # We load the selected model information
    with open(os.path.join("models", config["model_channels"], config["model_type"], "model_config.yaml"), "r") as config_buffer:
        model_config_file = yaml.safe_load(config_buffer)
    model_config = model_config_file.get("model_config")
    model = inference.ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_path, classifier_path)

    # We iterate over each set of images to process
    if os.path.exists("./path_list.csv"):
        path_list = open("./path_list.csv", 'r')
        for curr_set in path_list:

            if curr_set.strip() != "" and not curr_set.startswith("#"):
                curr_set_arr = curr_set.split(",")
                # We create the output folder
                os.makedirs(curr_set_arr[4].strip(), exist_ok=True)
                # We load the images as numpy arrays
                cell_data = []
                if "r" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[0].strip(), as_gray=True)])
                if "y" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[1].strip(), as_gray=True)])
                if "b" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[2].strip(), as_gray=True)])
                if "g" in config["model_channels"]:
                    cell_data.append([imread(curr_set_arr[3].strip(), as_gray=True)])

                inference.run_model(model, cell_data, os.path.join(curr_set_arr[4], curr_set_arr[5].strip()))
                config["log"].info("- Saved result for " +  os.path.join(curr_set_arr[4], curr_set_arr[5].strip()))

except Exception as e:
    config["log"].error("- " + str(e))

config["log"].info('----------')
config["log"].info('End: ' + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
