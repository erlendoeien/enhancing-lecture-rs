"""Script for running search on the best features and combinations of features
"""
import logging
import time
from functools import partial
from pathlib import Path
from pprint import pformat

import numpy as np
import optuna
import torch
from merlin_standard_lib import Schema, Tag
from transformers import EarlyStoppingCallback, set_seed
from transformers4rec import torch as tr
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

from evaluate_model import create_input_module_args, get_schema, parse_hyperparams
from hp_search import MODEL_TYPES, get_model, load_study, project_input
from utils.hyperparameter_search import (
    SEED,
    config_to_trial_param,
    empty_cache_cb,
    get_next_item_pred_task,
    save_sampler_cb,
)
from utils.io import load_json, setup_args, setup_logging
from utils.pre_processing import ITEM_COL

N_TRIALS = 200


def objective(
    trial: optuna.Trial,
    base_path: Path,
    out_path: Path,
    seed: int,
    model_type: MODEL_TYPES,
    schema: Schema,
    hyperparameters: dict,
):
    configs_path = Path("configs")

    feature_args_raw = config_to_trial_param(
        load_json(configs_path / f"{base_path.parent.parent}_features_config.json"),
        trial,
    )
    selected_features = [
        feat for feat, to_include in feature_args_raw.items() if to_include
    ]
    schema = schema.select_by_name([ITEM_COL] + selected_features)
    logging.debug(f"Subset schema:+\n{pformat(schema.column_names)}")

    # Get Optimal HP
    input_args, model_args, trainer_args = (
        hyperparameters["input_args"],
        hyperparameters["model_args"],
        hyperparameters["trainer_args"],
    )

    # Setting the seed for every trial
    set_seed(seed)
    # HARDCORDE FOR TESTING
    # model_config["d_model"] = 64
    # model_config["n_layer"] = 1
    # model_config["n_head"] = 2
    # input_config["embedding_dims"] = {ITEM_COL: 64}

    # Make sure input is projected into expected model size
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=model_args["total_seq_length"], **input_args
    )

    prediction_task = get_next_item_pred_task(hyperparameters["label_smoothing"])
    model = get_model(model_args, input_module, prediction_task, model_type)
    # HARDCODE FOR TESTING
    # training_args_config["per_device_train_batch_size"] = 64
    # training_args_config["gradient_accumulation_steps"] = 4
    # training_args_config["num_train_epochs"] = 1
    # training_args_config["disable_tqdm"] = False

    logging.debug(model)

    training_args = T4RecTrainingArguments(
        out_path / f"trial_{trial.number}",
        seed=seed,
        max_sequence_length=model_args["total_seq_length"],
        **trainer_args,
    )

    trainer = Trainer(
        model=model,
        schema=schema,
        args=training_args,
        train_dataset_or_path=str(base_path / "train.parquet"),
        eval_dataset_or_path=str(base_path / "val.parquet"),
        compute_metrics=True,
        # Num epochs tolerated with worsening and the threshold for NDCG@10
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.001
            )
        ],
    )

    trainer.train()
    eval_res = trainer.evaluate()
    # Returning the best found model and value
    return eval_res["eval_/next-item/ndcg_at_10"]


if __name__ == "__main__":
    parser = setup_args()
    args = vars(parser.parse_args())
    dataset, dataset_type, feature_set, log_name, schema_path, out_dir, model_type = (
        args["dataset"],
        args["dataset_type"],
        args["feature_set"],
        args["log_file"],
        args["schema_path"],
        args["out_dir"],
        args["model_type"],
    )
    log_dir = Path("logs")
    setup_logging(log_dir / log_name, level=logging.INFO)
    logging.info(f"GPU: {torch.cuda.get_device_name()}")
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")
    base_path = Path(dataset) / dataset_type / feature_set
    assert base_path.exists(), f"Dataset path {base_path} do not exist"

    schema = get_schema(schema_path, base_path)

    hyperparams = load_json(args["hyperparameter_path"])
    logging.info(f"Loaded hyperparameters: \n{pformat(hyperparams, indent=4)}")

    # Popping of hyper params which is related to loss or is a conditional HP
    label_smoothing = hyperparams.pop("label_smoothing", 0)
    trainable = hyperparams.pop("trainable", False)
    cont_project, agg_project = hyperparams.pop(
        "continuous_projection", False
    ), hyperparams.pop("aggregated_projection", False)

    input_args, model_args, trainer_args = parse_hyperparams(hyperparams)
    input_args["d_output"] = model_args["d_model"]
    if model_type == "gru":
        input_args["masking"] = "clm"
    input_module_args_raw = create_input_module_args(
        input_args, schema.column_names, is_trainable=trainable
    )
    input_module_args = project_input(
        input_module_args_raw,
        cont_project,
        agg_project,
        model_args["d_model"],
        len(schema.select_by_tag(Tag.CONTINUOUS)),
    )

    # Override static args
    trainer_args.update(
        per_device_eval_batch_size=trainer_args["per_device_train_batch_size"],
    )
    logging.info(f"Input module args:\n{pformat(input_module_args, indent=4)}")
    logging.info(f"Model args:\n{pformat(model_args, indent=4)}")
    logging.info(f"Training args:\n{pformat(trainer_args, indent=4)}")

    torch.cuda.empty_cache()

    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = Path(
            f"{dataset}_{model_type}_{dataset_type}_{feature_set}_{time.time()}"
        )
    out_path.mkdir(exist_ok=True)

    study_name = args["study_name"] or (
        f"{dataset}_features_{model_type}_" f"{np.random.random_integers(0, 1000)}"
    )
    relative_n_trials = N_TRIALS - args["num_trials_completed"]
    logging.info(f"Setting up study {study_name} with {relative_n_trials} trials")

    study = load_study(
        study_name,
        out_path,
        sampler=optuna.samplers.RandomSampler,
        study_db_name="hyperparameter_search.db",
    )

    logging.info(f"Study with sampler: {study.sampler}")

    trial_params = {
        "input_args": input_module_args,
        "model_args": model_args,
        "trainer_args": trainer_args,
        "label_smoothing": label_smoothing,
    }
    sampler_cb = partial(save_sampler_cb, path=out_path)
    study.optimize(
        func=lambda trial: objective(
            trial, base_path, out_path, SEED, model_type, schema, trial_params
        ),
        n_trials=relative_n_trials,  # ,N_TRIALS - args["num_trials_completed"],
        gc_after_trial=True,
        callbacks=[
            sampler_cb,
            empty_cache_cb,
        ],
        catch=[MemoryError],
    )
    logging.info(
        f"Found best parameters during trial \n{study.best_trial.number} "
        + f"with NDCG@10 at {study.best_value}"
    )
    logging.info(f"Best params:\n{study.best_params}")
    logging.info("COMPLETE")
