import logging
import os
import time
import warnings
from functools import partial
from pathlib import Path
from pprint import pformat

import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from optuna.exceptions import ExperimentalWarning
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from transformers import set_seed

from utils.constants import _MOOC_FLOAT_COLS, SEED  # create_skopt_sampler,
from utils.hyperparameter_search import (
    clean_out_dir_cb,
    config_to_trial_param,
    load_study,
    save_sampler_cb,
)
from utils.io import load_json, setup_args, setup_logging

# No convergence during max_iter search
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
# Warning for retry_failed_callback
warnings.filterwarnings(action="ignore", category=ExperimentalWarning)

N_TRIALS = 250
N_TASKS = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", 1))


# One objective function for each loss
SGD_LOSSES = ["hinge", "log_loss", "perceptron"]
K_FOLDS = 10
TEST_SIZE = 0.1
TARGET = "is_repeated"
EDNET_DEFAULT_FEATURES = [
    "time_spent",
    "num_forward",
    "num_backward",
    "num_pause",
    "median_pause",
    "time_comp",
    "time_played",
    "replay_length",
    "skipped_length",
    "seg_rep_60",
]
MOOC_DEFAULT_FEATURES = EDNET_DEFAULT_FEATURES + _MOOC_FLOAT_COLS


def get_model(model_name: str, model_args):
    if model_name in ["hinge", "log_loss", "perceptron"]:
        return SGDClassifier(model_name, **model_args)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(**model_args)
        # raise NotImplementedError("XGBoost is not supported yet")
    elif model_name == "random":
        return DummyClassifier(**model_args)
    else:
        raise RuntimeError(f"Can't get model for unknown model name {model_name}")


def get_model_args(trial: optuna.Trial, model_type: str):
    configs_path = Path("configs")
    if model_type in SGD_LOSSES:
        config = load_json(configs_path / "config_sgd.json")
        # Based Sklearn documentation for SGD convergence
        max_iter_guess = np.ceil(1e6 / X_train.shape[0])
        config["max_iter"] = {
            **config["max_iter"],
            "min": np.ceil(max_iter_guess * 0.5),
            "max": np.ceil(max_iter_guess * 1.5),
        }
    elif model_type == "xgboost":
        config = load_json(configs_path / "config_xgb.json")
        # config["gpu_id"] = -1
        # config["predictor"] = "cpu_predictor"
    else:
        raise RuntimeWarning(f"Not supported model {model_type}")

    model_args = config_to_trial_param(config, trial)
    if model_args.get("penalty", -1) == "elasticnet":
        # 0 is L2 and 1 is L1, which is already covered
        model_args["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9, step=0.1)

    return model_args


# def user_based_splitter(X: pd.DataFrame):


def objective(
    trial: optuna.Trial,
    X_trial: pd.DataFrame,
    y_trial: pd.Series,
    seed: int,
    model_type="hinge",
):
    # Load configs and generate hyperparameters
    model_args = get_model_args(trial, model_type)
    set_seed(seed)
    clf = get_model(model_type, model_args)
    if trial.number == 0:
        logging.info(pformat(clf.get_params(), indent=4))
    # Search with cv group by user
    acc = cross_val_score(
        clf,
        X_trial,
        y_trial,
        cv=GroupKFold(K_FOLDS),
        groups=X_trial.index.get_level_values("user_id").values,
        scoring="accuracy",
        n_jobs=N_TASKS,
    )
    return acc.mean()


def train_test_group_split(X, y, test_size=TEST_SIZE, random_state=SEED, **kwargs):
    # Stratified group split by user
    idx_train, idx_test = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        ).split(X, y, groups=X.index.get_level_values("user_id").values, **kwargs)
    )
    return (
        X.iloc[idx_train],
        X.iloc[idx_test],
        y.iloc[idx_train],
        y.iloc[idx_test],
    )


if __name__ == "__main__":
    parser = setup_args()
    args = vars(parser.parse_args())
    dataset, dataset_type, feature_set, log_name, out_dir = (
        args["dataset"],
        args["dataset_type"],
        args["feature_set"],
        args["log_file"],
        args["out_dir"],
    )
    log_dir = Path("logs")
    setup_logging(log_dir / log_name, level=logging.INFO)
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")
    base_path = Path(dataset) / dataset_type / feature_set
    assert base_path.exists(), f"Dataset path {base_path} do not exist"
    logging.info(
        "Loading dataset from "
        f"{(data_path:=base_path / 'repetition_bin_user_sampled.parquet')}"
    )
    data = pd.read_parquet(data_path)
    cont_features = args["continuous_float_features"] or (
        EDNET_DEFAULT_FEATURES if dataset == "ednet" else MOOC_DEFAULT_FEATURES
    )
    logging.info(
        f"Using selected features: \n{pformat(cont_features, indent=4, width=120)}"
    )
    # TODO: Add pipeline so categorical features can be supported
    # TODO: Only encode first of multi label
    subset_data = data[cont_features + ["is_repeated"]]

    X, y = (
        subset_data.drop(["is_repeated"], axis=1),
        subset_data.loc[:, "is_repeated"],
    )
    X_train, X_test, y_train, y_test = train_test_group_split(X, y)
    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = Path(
            f"{dataset}_{args['model_type']}_{dataset_type}_{feature_set}_{time.time()}"
        )
    out_path.mkdir(exist_ok=True)

    study_name = args["study_name"] or (
        f"rep_{dataset}_search_{args['model_type']}_"
        f"{np.random.random_integers(0, 1000)}"
    )
    relative_n_trials = N_TRIALS - args["num_trials_completed"]
    logging.info(f"Setting up study {study_name} with {relative_n_trials} trials")
    study = load_study(
        study_name,
        out_path,
        # sampler=create_skopt_sampler,
        study_db_name="rep_classification_search_fix.db",
    )
    logging.info(f"Study with sampler: {study.sampler}")
    sampler_cb = partial(save_sampler_cb, path=out_path)
    study.optimize(
        func=lambda trial: objective(
            trial,
            X_train,
            y_train,
            SEED,
            model_type=args["model_type"],
        ),
        n_trials=relative_n_trials,
        gc_after_trial=True,
        callbacks=[sampler_cb, lambda _, trial: clean_out_dir_cb(trial, out_path)],
        catch=[MemoryError],
    )
    logging.info(
        f"Found best parameters during trial \n{study.best_trial} "
        + f"with NDCG@10 at {study.best_value}"
    )
    logging.info(f"Best params:\n{study.best_params}")
    logging.info("COMPLETE")
