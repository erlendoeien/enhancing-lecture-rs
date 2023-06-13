# I/O UTILS
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Union

import numpy as np


def setup_logging(log_file, **kwargs):
    config = {
        **{
            "filename": log_file,
            "format": "%(asctime)s | %(levelname)s: %(message)s",
            "level": logging.DEBUG,
            "filemode": "w",
        },
        **kwargs,
    }
    logging.basicConfig(**config)


def save_enc(encoder: dict, path):
    with open(path, "w") as f:
        json.dump(encoder, f)


def save_embedding_table(dir_path, feature_name, embedding_table):
    """Store embedding tables as numpy matrices"""
    out_name = dir_path / f"session_{feature_name}.npy"
    with open(out_name, "wb") as file:
        np.save(file, embedding_table)


def load_embedding_table(dir_path, feature_name):
    """Load embedding tables as numpy matrices"""
    in_name = dir_path / f"session_{feature_name}.npy"
    with open(in_name, "rb") as file:
        emb_table = np.load(file)
    return emb_table


def load_json(json_path: Union[str, Path]):
    """Load file saved as JSON"""
    if json_path is None:
        raise RuntimeError("No path is provided")
    with open(json_path) as f:
        return json.load(f)


def setup_args():
    parser = argparse.ArgumentParser(
        description="General parser for loading datasets and optional hyper parameters"
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        default=f"tmp_log_{time.time()}.log",
        help="Log file name corresponding to the model-dataset combination",
    )
    parser.add_argument(
        "--feature-set",
        "-f",
        type=str,
        choices=[
            "all_scaled",
            "video_normalized",
            # "all_scaled_bias_adj", # DEPRECATED
            "bias_adj_all_scaled",
            "raw_dataset",
        ],
        default="all_scaled",
        help="Type of feature pre-processing",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=["ednet", "mooc"],
        default="mooc",
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--num-trials-completed",
        "-n",
        type=int,
        default=0,
        help="The number of trials already completed",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="The main output name to use, including dataset details",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=[
            "xlnet",
            "bert",
            "gru",
            "xgboost",
            "hinge",
            "log_loss",
            "random",
            "bpr",
            "als",
            "lmf",
            "knn",
        ],
        default="xlnet",
        help="Specify the model to use, by default XLNet",
    )
    parser.add_argument(
        "--dataset-type",
        "-m",
        type=str,
        choices=["sequential", "conventional"],
        default="sequential",
        help="Overall type of dataset",
    )
    parser.add_argument(
        "--schema-path",
        "-s",
        type=str,
        help="Path to schema file to use",
    )
    parser.add_argument(
        "--hyperparameter-path",
        "-p",
        type=str,
        help="Path to hyperparameter file to use. Supports only JSON",
    )
    parser.add_argument(
        "--study-name",
        "-S",
        type=str,
        help="The name of the study to use if doing hyperparameter search",
    )
    parser.add_argument(
        "--aggregated-projection",
        "-a",
        action="store_true",
        help="Flag to enable aggregated input projection",
    )
    parser.add_argument(
        "--continuous-projection",
        "-c",
        action="store_true",
        help="Flag to enable continuous input feature projection",
    )
    parser.add_argument(
        "--categorical-features",
        "-C",
        nargs="*",
        help="Categorical, OHE feature names to be used",
    )
    parser.add_argument(
        "--continuous-float-features",
        "-F",
        nargs="*",
        help="Continuous feature names in the float domain to be used",
    )
    return parser
