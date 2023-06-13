import logging
import os
import time
import warnings
from functools import partial
from pathlib import Path
from pprint import pformat

import implicit
import numpy as np
import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from scipy.sparse import csr_matrix, load_npz
from transformers import set_seed

from utils import SavePredictionsCallback
from utils.analysis import calculate_metrics
from utils.constants import SEED
from utils.hyperparameter_search import (
    clean_out_dir_cb,
    config_to_trial_param,
    load_study,
    save_sampler_cb,
)
from utils.io import load_json, setup_args, setup_logging
from utils.t4r_analysis import get_metrics

# Warning for retry_failed_callback
warnings.filterwarnings(action="ignore", category=ExperimentalWarning)

N_TRIALS = 150
N_TASKS = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", 1))


def get_model_config(model_name: str):  # , **model_args):
    if model_name.lower() == "als":
        return implicit.gpu.als.AlternatingLeastSquares  # (**model_args)
    elif model_name.lower() == "bpr":
        return implicit.gpu.bpr.BayesianPersonalizedRanking  # (**model_args)
    elif model_name.lower() == "lmf":
        return implicit.cpu.lmf.LogisticMatrixFactorization  # (**model_args)
    elif model_name.lower() == "cosine":
        return implicit.nearest_neighbours.CosineRecommender  # (**model_args)
    elif model_name.lower() == "tfidf":
        return implicit.nearest_neighbours.TFIDFRecommender  # (**model_args)
    elif model_name.lower() == "bm25":
        return implicit.nearest_neighbours.BM25Recommender  # (**model_args)
    else:
        raise RuntimeError(f"Can't get model for unknown model name {model_name}")


def get_model_args(trial: optuna.Trial, model_type: str, seed: int):
    configs_path = Path("configs")
    if model_type in ["als", "lmf"]:
        config_type = "mf"
    else:
        config_type = model_type
    config = load_json(configs_path / f"config_{config_type}.json")
    model_args = config_to_trial_param(config, trial)
    if (knn_type := model_args.pop("distance", None)) == "bm25":
        # Lower K1 -> Closer to jaccard distance, K1=0 -> Jaccard
        # Changing K1 mainly changes scale of weights, not shape
        # K1 -> Weight scaling
        # B -> effect of length normalization - B 0.5 -> Slight bias towards popularity
        # K =1.2 -> more important to match all terms, instead of repeated terms
        model_args["K1"] = trial.suggest_float("K1", 1e-7, 1e3, log=True)
        model_args["B"] = trial.suggest_float("B", 0, 1, step=0.1)
    if model_type == "knn":
        model_args["num_threads"] = N_TASKS
        model_type = knn_type
    else:
        model_args["random_state"] = seed

    return model_args, model_type


def predict(model, train_csr: csr_matrix, targets: torch.Tensor, N=20, save=False):
    # User_ids are mapped to each row for `train_csr` -> Arange is all users
    recs, scores = model.recommend(
        np.arange(train_csr.shape[0]), train_csr, N=N, filter_already_liked_items=False
    )
    metrics = get_metrics()
    storage_container = SavePredictionsCallback()
    preds = torch.tensor(recs)
    scores = torch.tensor(scores)
    labels = (preds == targets).int()
    for _, metric in metrics.items():
        metric.update(scores, labels)
    if save:
        storage_container(
            pred_item_ids=preds, pred_item_scores=scores, labels=targets.view(-1)
        )
    return metrics, storage_container


def objective(
    trial: optuna.Trial,
    X_trial: csr_matrix,
    y_trial: torch.Tensor,
    seed: int,
    model_type="als",
):
    # Load configs and generate hyperparameters
    model_args, model_name = get_model_args(trial, model_type, seed)
    set_seed(seed)
    logging.info(model_args)
    model = get_model_config(model_name)(**model_args)
    model.fit(X_trial)
    batch_metrics, _ = predict(model, X_trial, y_trial)
    metrics = calculate_metrics(batch_metrics)
    logging.info(pformat(metrics, indent=2))
    return metrics["ndcg"][10]


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
    if N_TASKS == 1:
        # GPU based job
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")

    base_path = Path(dataset) / dataset_type / feature_set
    assert base_path.exists(), f"Dataset path {base_path} do not exist"
    logging.info("Loading dataset from " f"{base_path}")
    train = load_npz(base_path / "train.npz").astype(float)
    val = load_npz(base_path / "val.npz").astype(float)
    targets = torch.tensor(val.indices).view(-1, 1)

    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = Path(
            f"{dataset}_{args['model_type']}_{dataset_type}_{feature_set}_{time.time()}"
        )
    out_path.mkdir(exist_ok=True)

    study_name = args["study_name"] or (
        f"rep_{dataset}_search_{args['model_type']}_" f"{np.random.randint(0, 1001)}"
    )
    relative_n_trials = N_TRIALS - args["num_trials_completed"]
    logging.info(f"Setting up study {study_name} with {relative_n_trials} trials")
    study = load_study(
        study_name,
        out_path,
        # sampler=create_skopt_sampler,
        study_db_name="baselines_search_fix.db",
    )
    logging.info(f"Study with sampler: {study.sampler}")
    sampler_cb = partial(save_sampler_cb, path=out_path)
    study.optimize(
        func=lambda trial: objective(
            trial,
            train,
            targets,
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
