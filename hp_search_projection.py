"""Script for running search on the best features and combinations of features
"""
import logging
import time
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Dict

import numpy as np
import optuna
import torch
from merlin_standard_lib import Schema, Tag
from transformers import EarlyStoppingCallback, set_seed
from transformers4rec import torch as tr
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer
from transformers4rec.torch.ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt

from evaluate_model import (
    create_input_module_args,
    get_schema,
    load_hp_config,
    parse_hyperparams,
)
from hp_search_xlnet import load_study
from utils.hyperparameter_search import (
    config_to_trial_param,
    empty_cache_cb,
    save_sampler_cb,
)
from utils.io import load_json, setup_args, setup_logging

MAX_SEQUENCE_LENGTH = 30
ITEM_COL = "item_id"
SEED = 379
HIDDEN_SIZE_MULT = 4
N_TRIALS = 4
N_EPOCHS = 50


def objective(
    trial: optuna.Trial,
    base_path: Path,
    out_path: Path,
    seed: int,
    schema: Schema,
    max_seq_len: int,
    hyperparameters: Dict[str, Dict],
):
    configs_path = Path("configs")
    projection_config = config_to_trial_param(
        load_json(configs_path / "projection_config.json"),
        trial,
    )

    # Get Optimal HP
    input_args, model_args, trainer_args = (
        hyperparameters["input_args"],
        hyperparameters["model_args"],
        hyperparameters["trainer_args"],
    )
    input_args = {**input_args}
    # Add shared continuous projection
    if projection_config["continuous_projection"]:
        num_cont_feats = len(schema.select_by_tag(Tag.CONTINUOUS))
        cont_vec_size = num_cont_feats * input_args["soft_embedding_dim_default"]
        input_args["continuous_projection"] = [
            cont_vec_size * HIDDEN_SIZE_MULT,
            cont_vec_size,
        ]

    # Add aggregated tensor projection
    if projection_config["aggregated_projection"]:
        input_args.pop("d_output")
        input_args["projection"] = tr.MLPBlock(
            [model_args["d_model"] * HIDDEN_SIZE_MULT, model_args["d_model"]]
        )

    # HARDCORDE FOR TESTING
    # model_config["d_model"] = 64
    # model_config["n_layer"] = 1
    # model_config["n_head"] = 2
    # input_config["embedding_dims"] = {ITEM_COL: 64}

    # Make sure input is projected into expected model size
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=max_seq_len, **input_args
    )

    transformer_config = tr.XLNetConfig.build(**model_args)

    metrics = [
        NDCGAt(top_ks=[5, 10], labels_onehot=True),
        RecallAt(top_ks=[5, 10], labels_onehot=True),
        AvgPrecisionAt(top_ks=[5, 10], labels_onehot=True),
    ]
    label_smoothing_xe_loss = tr.LabelSmoothCrossEntropyLoss(
        reduction="mean", smoothing=hyperparameters["label_smoothing"]
    )

    prediction_task = tr.NextItemPredictionTask(
        weight_tying=True, metrics=metrics, loss=label_smoothing_xe_loss
    )
    model = transformer_config.to_torch_model(input_module, prediction_task)
    # HARDCODE FOR TESTING
    # training_args_config["per_device_train_batch_size"] = 64
    # training_args_config["gradient_accumulation_steps"] = 4
    # training_args_config["num_train_epochs"] = 1
    # training_args_config["disable_tqdm"] = False

    logging.info(model)

    training_args = T4RecTrainingArguments(
        out_path / f"trial_{trial.number}",
        seed=seed,
        max_sequence_length=max_seq_len,
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
    dataset, dataset_type, feature_set, log_name, schema_path, out_dir = (
        args["dataset"],
        args["dataset_type"],
        args["feature_set"],
        args["log_file"],
        args["schema_path"],
        args["out_dir"],
    )
    log_dir = Path("logs")
    setup_logging(log_dir / log_name, level=logging.INFO)
    logging.info(f"GPU: {torch.cuda.get_device_name()}")
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")
    base_path = Path(dataset) / dataset_type / feature_set
    assert base_path.exists(), f"Dataset path {base_path} do not exist"

    base_schema = get_schema(schema_path, base_path)

    logging.info("Filtering schema")
    schema = base_schema.select_by_name(
        [ITEM_COL]
        + (args["continuous_int_features"] or [])
        + (args["categorical_features"] or [])
        + (args["continuous_float_features"] or [])
    )
    logging.info(f"Schema:\n{pformat(schema.to_dict(), indent=4)}")

    hyperparams = load_hp_config(args["hyperparameter_path"])
    logging.info(f"Loaded hyperparameters: \n{pformat(hyperparams, indent=4)}")

    label_smoothing = hyperparams.pop("label_smoothing", 0)
    trainable = hyperparams.pop("trainable", False)

    input_args, model_args, trainer_args = parse_hyperparams(hyperparams)
    input_args["d_output"] = model_args["d_model"]
    input_module_args = create_input_module_args(
        input_args, schema.column_names, is_trainable=trainable
    )

    # Override static args
    trainer_args.update(
        per_device_eval_batch_size=trainer_args["per_device_train_batch_size"],
        num_train_epochs=N_EPOCHS,
        report_to=[],
    )
    logging.info(f"Input module args:\n{pformat(input_module_args, indent=4)}")
    logging.info(f"Model args:\n{pformat(model_args, indent=4)}")
    logging.info(f"Training args:\n{pformat(trainer_args, indent=4)}")

    set_seed(SEED)
    torch.cuda.empty_cache()

    if out_dir:
        out_path = Path(out_dir)
    else:
        # TODO: ADJUST to add some feature information in out directory name
        out_path = Path(f"{dataset}_{dataset_type}_{feature_set}_{time.time()}")
    out_path.mkdir(exist_ok=True)

    study_name = (
        args["study_name"]
        or f"{dataset}_projection_{np.random.random_integers(0, 1000)}"
    )
    logging.info(f"Setting up study {study_name} with {N_TRIALS} trials")
    search_space = load_json("configs/projection_config.json")
    logging.info(f"Search space: \n{pformat(search_space)}")
    study = load_study(
        study_name,
        out_path,
        sampler=optuna.samplers.GridSampler,
        search_space=search_space,
    )

    trial_params = {
        "input_args": input_module_args,
        "model_args": model_args,
        "trainer_args": trainer_args,
        "label_smoothing": label_smoothing,
    }
    sampler_cb = partial(save_sampler_cb, path=out_path)
    study.optimize(
        func=lambda trial: objective(
            trial, base_path, out_path, SEED, schema, MAX_SEQUENCE_LENGTH, trial_params
        ),
        n_trials=N_TRIALS,
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
    logging.info(
        f"Found best parameters during trial \n{study.best_trial.number} "
        + f"with NDCG@10 at {study.best_value}"
    )
    logging.info(f"Best params:\n{study.best_params}")
    logging.info("COMPLETE")
