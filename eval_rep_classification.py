import logging
import pickle
import time
import warnings
from pathlib import Path
from pprint import pformat

import pandas as pd
import torch
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix

from hp_rep_classification import (
    EDNET_DEFAULT_FEATURES,
    MOOC_DEFAULT_FEATURES,
    SGD_LOSSES,
    TARGET,
    TEST_SIZE,
    get_model,
    train_test_group_split,
)
from utils.constants import SEED
from utils.io import load_json, setup_args, setup_logging

# No convergence during max_iter search
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
# Warning for retry_failed_callback
warnings.filterwarnings(action="ignore", category=ExperimentalWarning)


def get_model_args(model_type: str, seed: int):
    configs_path = Path("configs")
    if model_type in SGD_LOSSES:
        config = load_json(configs_path / "config_sgd.json")
        config["random_state"] = seed
    elif model_type == "xgboost":
        config = load_json(configs_path / "config_xgb.json")
        config["seed"] = seed
    elif model_type == "random":
        config = load_json(configs_path / "config_random.json")
        config["random_state"] = seed
    else:
        raise RuntimeWarning(f"Not supported model {model_type}")

    # Only return scalar values
    return {
        key: value for key, value in config.items() if type(value) not in [dict, list]
    }


def predict(clf, X_test):
    y_pred = clf.predict(X_test)
    if model_type == "hinge":
        y_proba = clf.decision_function(X_test)
    else:
        y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def gen_reports(y_test, y_pred):
    logging.info("\tClassification report balanced:")
    class_report = classification_report(y_test, y_pred)
    logging.info(f"\n{class_report}")

    logging.info("\tConfusion matrix balanced")
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info(f"\n{conf_matrix}")


def save_evaluation(
    clf,
    X_test,
    y_test,
    y_pred,
    y_proba,
    out_path,
    seed=SEED,
    prediction_name=f"seed_{SEED}",
):
    logging.info("\tSaving model")
    model_out = out_path / "models"
    model_out.mkdir(parents=True, exist_ok=True)
    with open(model_out / f"{seed}.pickle", "wb") as f:
        pickle.dump(clf, f)

    predict_path = out_path / "predictions"
    predict_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"\tSaving prediction data to {predict_path}")
    predictions_df = pd.concat(
        [
            X_test.reset_index(),
            y_test.reset_index(),
            pd.Series(y_pred, name="y_pred"),
            pd.Series(y_proba, name="y_proba"),
        ],
        axis=1,
    )
    print(predictions_df.head())
    print(predictions_df.dtypes)
    print(y_pred)
    predictions_df = predictions_df.loc[:, ~predictions_df.columns.duplicated()]
    predictions_df.to_parquet(predict_path / f"{prediction_name}.parquet")
    return predictions_df


def sample_majority(df, binary_labels=[True, False]):
    unbalanced_class_dist = df_full[TARGET].value_counts()
    logging.info(
        f"\tUnbalanced distribution: \n{unbalanced_class_dist / df_full.shape[0]}"
    )
    minority_label, majority_label = binary_labels
    # repetitions are minority -> Get the ratio to downsample
    desired_ratio = (
        unbalanced_class_dist[minority_label] / unbalanced_class_dist[majority_label]
    )
    num_minority = df[TARGET].sum()
    majority_sample_n = int(num_minority / desired_ratio)
    logging.info(f"\tSampling {majority_sample_n} of non-repeat interactions")

    logging.info(f"\tBefore sampling: {df.shape}")
    logging.info(f"\tFull df class distribution: \n{df.value_counts(TARGET)}")
    full_test_minority = df[df[TARGET] == minority_label]
    print(df[df[TARGET] == majority_label])
    full_test_majority = df[df[TARGET] == majority_label].sample(
        n=majority_sample_n, random_state=SEED
    )
    sampled = pd.concat([full_test_minority, full_test_majority])
    logging.info(f"\tSampled class distribution: \n{sampled.value_counts(TARGET)}")
    return sampled


if __name__ == "__main__":
    parser = setup_args()
    args = vars(parser.parse_args())
    dataset, dataset_type, feature_set, model_type, log_name, out_dir = (
        args["dataset"],
        args["dataset_type"],
        args["feature_set"],
        args["model_type"],
        args["log_file"],
        args["out_dir"],
    )
    log_dir = Path("logs")
    setup_logging(log_dir / log_name, level=logging.INFO)
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")
    hyperparams = load_json(args["hyperparameter_path"])
    logging.info(f"Loaded hyperparameters: \n{pformat(hyperparams, indent=4)}")

    base_path = Path(dataset) / dataset_type / feature_set
    assert base_path.exists(), f"Dataset path {base_path} do not exist"
    logging.info(
        "Loading dataset from "
        f"{(data_path:=base_path / 'repetition_bin_user_sampled_with_session_id.parquet')}"
    )
    data = pd.read_parquet(data_path)

    cont_features = args["continuous_float_features"] or (
        EDNET_DEFAULT_FEATURES if dataset == "ednet" else MOOC_DEFAULT_FEATURES
    )
    logging.info(f"Using selected features: \n{pformat(cont_features, indent=4)}")
    # TODO: Add pipeline so categorical features can be supported
    subset_data = data[cont_features + [TARGET]]
    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = Path(
            f"rep_{dataset}_{model_type}_{dataset_type}_{feature_set}_{time.time()}"
        )
    out_path.mkdir(exist_ok=True)

    X, y = (
        subset_data.drop([TARGET], axis=1),
        subset_data.loc[:, TARGET],
    )
    logging.info("Splitting balanced data")
    X_train, X_test, y_train, y_test = train_test_group_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    logging.info("Getting untouched, unbalanced dataset")
    df_full = pd.read_parquet(
        base_path / "is_repeated.parquet", columns=cont_features + [TARGET]
    )
    logging.info(f"X_train: {X_train.shape}")
    logging.info(f"Full {df_full.shape}")
    logging.info("Dropping seen interactions")
    logging.info(
        f"Num seen interactions: {df_full.index.intersection(X_train.index).shape}"
    )
    df_full_test = df_full[~df_full.index.isin(X_train.index)]
    logging.info("Sampling to original unbalanced distribution")
    full_test_sampled = sample_majority(df_full_test)

    X_test_full, y_test_full = full_test_sampled.drop(
        [TARGET], axis=1
    ), full_test_sampled[TARGET].astype(bool)

    # Overwrite config with found hyperparams
    model_args = {**get_model_args(model_type, SEED), **hyperparams}
    logging.info(f"Model args: \n{pformat(model_args, indent=4)}")
    clf = get_model(model_type, model_args)
    logging.info("Fitting model")
    clf.fit(X_train, y_train)

    logging.info("Evaluating model on balanced test")
    y_pred, y_proba = predict(clf, X_test)
    gen_reports(y_test, y_pred)
    save_evaluation(clf, X_test, y_test, y_pred, y_proba, out_path)

    logging.info("Evaluating model on full, unbalanced test set")
    y_pred_full, y_proba_full = predict(clf, X_test_full)
    gen_reports(y_test_full, y_pred_full)
    save_evaluation(
        clf,
        X_test_full,
        y_test_full,
        y_pred_full,
        y_proba_full,
        out_path,
        prediction_name=f"seed_{SEED}_full",
    )

    logging.info("COMPLETE")
