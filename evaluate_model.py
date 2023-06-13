import json
import logging
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import torch
from merlin_standard_lib import Tag
from transformers import EarlyStoppingCallback, set_seed
from transformers4rec import torch as tr
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

from hp_search import get_model, project_input
from utils import SavePredictionsCallback
from utils.constants import ITEM_COL, MAX_SEQUENCE_LENGTH, SEED
from utils.hyperparameter_search import parse_hyperparams
from utils.io import load_json, setup_args, setup_logging
from utils.schema import get_schema
from utils.t4r_utils import create_input_module_args, get_next_item_pred_task

NUM_TRAIN_EPOCHS = 75
NUM_SEEDS = 10


def store_metrics(results: dict, out_path: Path):
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()
    results_path = results_dir / out_path.name
    json_out = results_dir / f"{out_path.name}.json"
    parquet_out = results_dir / f"{out_path.name}.parquet"
    # Append to if exists
    results_df = pd.DataFrame.from_dict(results, orient="index")

    if json_out.exists() and parquet_out.exists():
        logging.info("Adding results to existing ones")
        results_df = pd.concat([pd.read_parquet(parquet_out), results_df])

    logging.info(f"Storing metrics to {results_path}.<json|parquet>")
    results_df.to_parquet(parquet_out)
    results_df.to_json(json_out, indent=4, orient="index")


def train_model(
    input_module_args: dict,
    model_args: dict,
    trainer_args: dict,
    base_path: Path,
    out_path_base: Path,
    seed: int,
    label_smoothing: float,
    test: bool = False,
):
    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        **input_module_args,
    )

    prediction_task = get_next_item_pred_task(label_smoothing)

    model = get_model(model_args, input_module, prediction_task, model_type)

    model_out_path = out_path_base / (
        f"seed_{seed}" if test else f"epoch_hp_seed_{seed}"
    )
    training_args = T4RecTrainingArguments(
        str(model_out_path),
        seed=seed,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        **trainer_args,
        per_device_eval_batch_size=trainer_args["per_device_train_batch_size"],
    )

    train_path = str(base_path / "train.parquet")
    val_path = str(base_path / "val.parquet")
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)
    ]
    if test:
        train_path = str(base_path / "val.parquet")
        val_path = str(base_path / "test.parquet")
        callbacks = None

    # Train on both train and val? Can one then use test for validation?
    trainer = Trainer(
        model=model,
        schema=schema,
        args=training_args,
        train_dataset_or_path=train_path,
        eval_dataset_or_path=val_path,
        compute_metrics=True,
        callbacks=callbacks,
    )
    return trainer, model


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
    base_schema = get_schema(schema_path, base_path)

    logging.info("Filtering schema")
    schema = base_schema.select_by_name(
        [ITEM_COL]
        + (args["categorical_features"] or [])
        + (args["continuous_float_features"] or [])
    )
    logging.info(f"Schema:\n{pformat(schema.to_dict(), indent=4)}")

    hyperparams = load_json(args["hyperparameter_path"])
    logging.info(f"Loaded hyperparameters: \n{pformat(hyperparams, indent=4)}")
    # Popping of hyper params which is related to loss or is a conditional HP
    label_smoothing = hyperparams.pop("label_smoothing", 0)
    is_trainable = hyperparams.pop("trainable", False)
    cont_project, agg_project = hyperparams.pop(
        "continuous_projection", False
    ), hyperparams.pop("aggregated_projection", False)

    input_args, model_args, trainer_args = parse_hyperparams(
        hyperparams, is_item_only=len(schema.column_names) == 1
    )
    input_args["d_output"] = model_args["d_model"]  # model_d_model
    if model_type == "gru":
        input_args["masking"] = "clm"
    # Remove Infer Embedding Sizes Multiplier since all categorical features
    # are pretrained
    if input_args.get("pretrained_embeddings", False):
        input_args.pop("infer_embedding_sizes_multiplier", None)
    input_module_args_raw = create_input_module_args(
        input_args, schema.column_names, is_trainable=is_trainable
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
        # disable_tqdm=False,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        report_to=["tensorboard"],
        predict_top_k=20,
    )

    logging.info(f"Input module args:\n{pformat(input_module_args, indent=4)}")
    logging.info(f"Model args:\n{pformat(model_args, indent=4)}")
    logging.info(f"Training args:\n{pformat(trainer_args, indent=4)}")

    out_path_base = Path(out_dir)
    out_path_base.mkdir(exist_ok=True)
    predictions_base_path = Path("predictions")
    train_seeds = np.random.default_rng(SEED).integers(1, 10000, NUM_SEEDS)
    already_completed_seeds = args["num_trials_completed"]

    logging.info("Getting optimal epoch num")

    optimal_epoch_out = out_path_base / f"epoch_hp_state_seed_{train_seeds[0]}.json"
    if optimal_epoch_out.exists():
        logging.info("\tLoading already found num epochs")
        with open(optimal_epoch_out) as f:
            epoch_state = json.load(f)
        NUM_EVAL_EPOCHS = int(epoch_state["epoch"])
    else:
        if already_completed_seeds > 0:
            raise RuntimeError(
                f"About to find optimal epoch, but said already completed {already_completed_seeds}"
            )
        logging.info("First time, EPOCH training")
        epoch_trainer, epoch_model = train_model(
            input_module_args,
            model_args,
            trainer_args,
            base_path,
            out_path_base,
            train_seeds.tolist()[0],
            label_smoothing,
            test=False,
        )
        epoch_trainer.train()
        logging.info(f"Optimal epoch state: {epoch_trainer.state}")
        NUM_EVAL_EPOCHS = int(epoch_trainer.state.epoch)
        logging.info("Saving optimal epoch state")
        epoch_trainer.state.save_to_json(optimal_epoch_out)
        del epoch_trainer
        del epoch_model

    results = {}

    logging.info("Updating trainer args with eval args")
    trainer_args.update(
        # disable_tqdm=False,
        num_train_epochs=NUM_EVAL_EPOCHS,
        report_to=["tensorboard"],
        predict_top_k=100,
        load_best_model_at_end=False,
        # evaluation_strategy="no",
    )

    # Training loop - Skip loaded ones
    seed_enumerator = enumerate(
        train_seeds[already_completed_seeds:].tolist(), start=already_completed_seeds
    )
    for idx, seed in seed_enumerator:
        torch.cuda.empty_cache()
        logging.info(f"Training {idx} with seed {seed}")
        set_seed(seed)
        # No additional projection/calculation of input
        # TODO: No load best model at end
        # TODO: Optimal num epochs changed
        test_trainer, test_model = train_model(
            input_module_args,
            model_args,
            trainer_args,
            base_path,
            out_path_base,
            seed,
            label_smoothing,
            test=True,
        )
        logging.info("TRAINING")
        test_trainer.train()

        # Setup prediction save callbacks and paths
        # TODO: Rewrite to a TrainerCallback and append to trainer.args.callbacks
        # after training
        log_preds = SavePredictionsCallback()
        test_trainer.log_predictions_callback = log_preds
        test_trainer.args.log_predictions = True
        # trainer.eval_dataset_or_path = str(base_path / "test.parquet")
        logging.info("PREDICT")
        predict_res = test_trainer.evaluate()
        logging.info(f"\n{pformat(predict_res, indent=4)}")

        predict_path = predictions_base_path / out_dir
        predict_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving prediction data to {predict_path}")
        log_preds.predictions_to_parquet(predict_path / f"seed_{seed}.parquet")
        # seed_results = {"eval": eval_res, "predict": predict_res}
        results[seed] = predict_res
    store_metrics(results, out_path_base)
    logging.info("COMPLETE")
