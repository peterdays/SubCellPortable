import os
from urllib.parse import urlparse

import boto3
import requests
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from subcellportable.vit_model import ViTPoolClassifier

# to avoid relative path problems
base_path = os.path.dirname(os.path.abspath(__file__))

def model_load(config, temp_path_suffix = None):
    # We load the selected model information
    with open(
        os.path.join(
            base_path,
            "models",
            config["model_channels"],
            config["model_type"],
            "model_config.yaml",
        ),
        "r",
    ) as config_buffer:
        model_config_file = yaml.safe_load(config_buffer)

    classifier_paths = None
    if "classifier_paths" in model_config_file:
        classifier_paths = model_config_file["classifier_paths"]
    classifier_paths = [os.path.join(base_path, p) for p in classifier_paths]

    encoder_path = model_config_file["encoder_path"]
    print("AAAAAAAAAAA")
    print(classifier_paths)
    print()
    if temp_path_suffix is not None:
        # ugly workaround: to guarantee that triggering multiple downloads doesnt break
        encoder_path = encoder_path.split(".")[0] + temp_path_suffix + "." + encoder_path.split(".")[1]
        classifier_paths = [p.split(".")[0] + temp_path_suffix + "." + p.split(".")[1] for p in classifier_paths]
    print("AAAAAAAAAAA")
    print(classifier_paths)
    print()
    encoder_path = os.path.join(base_path, encoder_path)

    needs_update = config["update_model"]
    for curr_classifier in classifier_paths:
        needs_update = needs_update or not os.path.isfile(curr_classifier)
    needs_update = needs_update or not os.path.isfile(encoder_path)

    # Checking for model update
    if needs_update:
        config["log"].info("- Downloading models...")
        with open(os.path.join(base_path, "models_urls.yaml"), "r") as urls_file:
            url_info = yaml.safe_load(urls_file)

            for index, curr_url_info in enumerate(url_info[config["model_channels"]][config["model_type"]]["classifiers"]):
                if curr_url_info.startswith("s3://"):
                    # try:
                    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                    urlcomponents = urlparse(curr_url_info)
                    print(classifier_paths[index], "\n", urlcomponents.path[1:])
                    s3.download_file(urlcomponents.netloc, urlcomponents.path[1:], classifier_paths[index])
                    config["log"].info("  - " + classifier_paths[index] + " updated.")
                    # except ClientError as e:
                    #     config["log"].warning("  - " + classifier_paths[index] + " s3 url " + curr_url_info + " not working.")
                else:
                    response = requests.get(curr_url_info)
                    if response.status_code == 200:
                        with open(classifier_paths[index], "wb") as downloaded_file:
                            downloaded_file.write(response.content)
                        config["log"].info("  - " + classifier_paths[index] + " updated.")
                    else:
                        config["log"].warning("  - " + classifier_paths[index] + " url " + curr_url_info + " not found.")

            curr_url_info = url_info[config["model_channels"]][config["model_type"]]["encoder"]
            if curr_url_info.startswith("s3://"):
                try:
                    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                    urlcomponents = urlparse(curr_url_info)
                    s3.download_file(urlcomponents.netloc, urlcomponents.path[1:], encoder_path)
                    config["log"].info("  - " + encoder_path + " updated.")
                except ClientError as e:
                    config["log"].warning("  - " + encoder_path + " s3 url " + curr_url_info + " not working.")
            else:
                response = requests.get(curr_url_info)
                if response.status_code == 200:
                    with open(encoder_path, "wb") as downloaded_file:
                        downloaded_file.write(response.content)
                    config["log"].info("  - " + encoder_path + " updated.")
                else:
                    config["log"].warning("  - " + encoder_path + " url " + curr_url_info + " not found.")

    model_config = model_config_file.get("model_config")
    model = ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_path, classifier_paths)

    return model
