import logging
import time
import warnings
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from optuna.exceptions import ExperimentalWarning
from scipy.sparse import load_npz
from transformers import set_seed

from evaluate_model import store_metrics
from hp_baselines import get_model_config, predict
from utils.analysis import calculate_metrics
from utils.constants import SEED
from utils.io import load_json, setup_args, setup_logging

# Warning for retry_failed_callback
warnings.filterwarnings(action="ignore", category=ExperimentalWarning)

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
    logging.info(f"GPU: {torch.cuda.get_device_name()}")
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")

    base_path = Path(dataset) / dataset_type / feature_set
    assert base_path.exists(), f"Dataset path {base_path} do not exist"

    logging.info("Loading dataset from " f"{base_path}")
    # train = load_npz(base_path / "train.npz").astype(float)
    # TODO: Replace to train with full train set, incl. validation data
    train = load_npz(base_path / "val_full.npz").astype(float)
    val = load_npz(base_path / "test.npz").astype(float)
    targets = torch.tensor(val.indices).view(-1, 1)

    hyperparams = load_json(args["hyperparameter_path"])
    logging.info(f"Loaded hyperparameters: \n{pformat(hyperparams, indent=4)}")
    model_name = model_type
    if model_type == "knn":
        model_name = hyperparams.pop("distance")

    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = Path(
            f"{dataset}_{model_type}_{dataset_type}_{feature_set}_{time.time()}"
        )
    out_path.mkdir(exist_ok=True)
    predictions_path = Path("predictions")
    predictions_path.mkdir(exist_ok=True)
    model_out = out_path / "models"
    model_out.mkdir(parents=True, exist_ok=True)

    NUM_SEEDS = 10
    train_seeds = np.random.default_rng(SEED).integers(1, 10000, NUM_SEEDS)
    already_completed_seeds = args["num_trials_completed"]
    seed_enumerator = enumerate(
        train_seeds[already_completed_seeds:].tolist(), start=already_completed_seeds
    )
    seed_results = {}
    for idx, seed in seed_enumerator:
        # Overwrite config with found hyperparams
        logging.info(f"Training {idx} with seed {seed}")

        model_args = {**hyperparams}
        if model_type != "knn":
            model_args["random_state"] = seed
        set_seed(seed)
        model = get_model_config(model_name)(**model_args)

        logging.info("TRAINING")
        model.fit(train)

        logging.info("PREDICT")
        batch_metrics, storage = predict(model, train, targets, save=True)

        metrics = calculate_metrics(batch_metrics)
        seed_results[seed] = {
            metric: {str(cut_off): val for cut_off, val in inner_dict.items()}
            for metric, inner_dict in metrics.items()
        }

        logging.info("\n" + pformat(metrics, indent=2))

        logging.info("Saving model")
        model.save(model_out / f"{seed}.npz")

        pred_out_dir = predictions_path / out_dir
        pred_out_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Saving prediction data to {pred_out_dir}")
        storage.predictions_to_parquet(pred_out_dir / f"seed_{seed}.parquet")
    # Map int keys to str
    store_metrics(seed_results, out_path)
    logging.info("COMPLETE")
