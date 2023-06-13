import logging
import time
import typing as t
from functools import partial
from pathlib import Path
from pprint import pformat

import numpy as np
import optuna
import torch
from merlin_standard_lib import Schema, Tag
from torch.cuda import empty_cache
from transformers import EarlyStoppingCallback, set_seed
from transformers4rec import config as t4r_config
from transformers4rec import torch as tr
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

from utils.constants import ITEM_COL, SEED
from utils.hyperparameter_search import (
    config_to_trial_param,
    load_study,
    save_sampler_cb,
)
from utils.io import load_json, setup_args, setup_logging
from utils.schema import get_schema
from utils.t4r_utils import create_input_module_args, get_next_item_pred_task

N_TRIALS = 150
# N_EPOCHS = 20


TRANFORMER_TYPES = t.Literal[
    "xlnet",
    "bert",
]
MODEL_TYPES = t.Literal[TRANFORMER_TYPES, "gru"]


def empty_cache_cb(study_, _):
    empty_cache()


def clean_out_dir_cb(trial: optuna.trial.FrozenTrial, out_path: Path):
    "Clean up trials to avoid data overload"
    trial_path = out_path / f"trial_{trial.number}"
    logging.info(f"Cleaning up {trial_path}")
    if trial_path.exists() and trial_path.is_dir():
        for chckpt_dir in trial_path.iterdir():
            if chckpt_dir.is_dir():
                for file in chckpt_dir.iterdir():
                    if file.stem != "trainer_state":
                        file.unlink()


def maybe_project(
    trial: optuna.Trial, maybe_cont_project: bool, maybe_agg_project: bool
):
    """Conditionally apply continuous or aggregated projection to input tensors"""
    cont_project, agg_project = False, False
    if maybe_cont_project:
        cont_project = trial.suggest_categorical("continuous_projection", [False, True])
    if maybe_agg_project:
        agg_project = trial.suggest_categorical("aggregated_projection", [False, True])
    return cont_project, agg_project


def project_input(
    input_args: t.Dict,
    cont_project: bool,
    agg_project: bool,
    d_model: int,
    num_cont_feats: int,
    hidden_size_multiplier=4,
) -> t.Dict[str, t.Union[str, int]]:
    """Add optional continuous or aggregated projection
    to inputs"""
    args = {**input_args}
    if cont_project:
        # If not SOHE -> Scalar features
        cont_vec_size = num_cont_feats * args.get("soft_embedding_dim_default", 1)
        args["continuous_projection"] = [
            cont_vec_size * hidden_size_multiplier,
            cont_vec_size,
        ]

    # Add aggregated tensor projection
    if agg_project:
        args.pop("d_output")
        args["projection"] = tr.MLPBlock([d_model * hidden_size_multiplier, d_model])
    return args


def get_transformer_config(
    model_type: MODEL_TYPES, model_args: t.Dict
) -> tr.T4RecConfig:
    if model_type == "xlnet":
        return tr.XLNetConfig.build(**model_args)
    elif model_type == "bert":
        return t4r_config.transformer.BertConfig.build(**model_args)
    else:
        raise RuntimeError(f"Currently locally unsupported config {model_type}")


def get_gru_block(model_config: t.Dict):
    """Create a GRU-block based on a model config"""
    return tr.Block(
        torch.nn.GRU(
            input_size=model_config["d_model"],
            hidden_size=model_config["d_model"],
            num_layers=model_config["n_layer"],
            dropout=model_config["dropout"],
        ),
        # Block will infer batch size -> Pass None
        [None, model_config["total_seq_length"], model_config["d_model"]],
    )


def get_model(model_args: t.Dict, input_module, prediction_task, model_type: str):
    if model_type == "gru":
        # Setup custom body and model,
        body = tr.SequentialBlock(input_module, get_gru_block(model_args))
        return tr.Model(tr.Head(body, prediction_task))
    return get_transformer_config(model_type, model_args).to_torch_model(
        input_module, prediction_task
    )


def objective(
    trial: optuna.Trial,
    base_path: Path,
    out_path: Path,
    seed: int,
    schema: Schema,
    continuous_projection: bool,
    aggregated_projection: bool,
    model_type="xlnet",
):
    # Load configs and generate hyperparameters
    configs_path = Path("configs")
    raw_input_config = load_json(configs_path / "config_input.json")
    raw_model_config = load_json(configs_path / "config_model.json")
    side_info_raw_config = {}
    # Only laod side info if needed
    is_item_only = len(schema.column_names) == 1 and schema.column_names[0] == ITEM_COL

    if not is_item_only:
        side_info_raw_config.update(
            load_json(configs_path / "config_side_info_input.json")
        )

    # Reduce Search space
    if model_type == "gru":
        raw_model_config.pop("n_head")
        raw_model_config.pop("layer_norm_eps")
        raw_input_config.pop("mlm_probability")
        raw_input_config["masking"] = "clm"

    input_config = config_to_trial_param(raw_input_config, trial)

    # Skip pretrained embeddings for datasets without ones
    pretrained_embeddings_opts = side_info_raw_config.pop(
        "pretrained_embeddings", False
    )
    is_trainable = False
    if base_path.parent.parent.name == "mooc" and not is_item_only:
        pretrained_embeddings = trial.suggest_categorical(
            "pretrained_embeddings", pretrained_embeddings_opts
        )
        side_info_raw_config["pretrained_embeddings"] = pretrained_embeddings
        if pretrained_embeddings:
            is_trainable = trial.suggest_categorical("trainable", [False, True])
            # Remove infer_embeddinge size multiplier if pretrained since all non-item
            #  categorical values have pretrained embeddings
            side_info_raw_config.pop("infer_embedding_sizes_multiplier", None)

    side_info_input_config = config_to_trial_param(side_info_raw_config, trial)

    model_config = config_to_trial_param(raw_model_config, trial)
    training_args_config = config_to_trial_param(
        load_json(configs_path / "config_trainer_args.json"), trial
    )
    do_cont_project, do_agg_project = maybe_project(
        trial, continuous_projection, aggregated_projection
    )

    # Create input module
    # Make sure input is projected into expected model size
    input_config["d_output"] = model_config["d_model"]
    input_config.update(side_info_input_config)
    input_module_args_raw = create_input_module_args(
        input_config, schema.column_names, is_trainable=bool(is_trainable)
    )
    input_module_args = project_input(
        input_module_args_raw,
        do_cont_project,
        do_agg_project,
        model_config["d_model"],
        len(schema.select_by_tag(Tag.CONTINUOUS)),
    )

    # Setting the seed for every trial - Samplers are independent
    set_seed(seed)
    # logging.info(f"Input module args: {pformat(input_module_args, indent=4)}")

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=model_config["total_seq_length"],
        **input_module_args,
    )

    label_smoothing = trial.suggest_float("label_smoothing", 0, 0.9, step=0.1)

    prediction_task = get_next_item_pred_task(label_smoothing)
    model = get_model(model_config, input_module, prediction_task, model_type)

    # TESTING
    # training_args_config["per_device_train_batch_size"] = 256
    # training_args_config["gradient_accumulation_steps"] = 2
    # training_args_config["num_train_epochs"] = 1
    # training_args_config["disable_tqdm"] = False
    # training_args_config.update(num_train_epochs=N_EPOCHS)
    logging.debug(model)
    logging.debug(input_module_args)
    logging.debug(training_args_config)

    training_args = T4RecTrainingArguments(
        out_path / f"trial_{trial.number}",
        seed=seed,
        max_sequence_length=model_config["total_seq_length"],
        **training_args_config,
        per_device_eval_batch_size=max(
            64, training_args_config["per_device_train_batch_size"] // 2
        ),
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
    torch.cuda.empty_cache()
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
        # + (args["continuous_int_features"] or [])
        + (args["categorical_features"] or [])
        + (args["continuous_float_features"] or [])
    )
    logging.info(f"Schema:\n{pformat(schema.to_dict(), indent=4)}")

    torch.cuda.empty_cache()

    if out_dir:
        out_path = Path(out_dir)
    else:
        # TODO: ADJUST to add some feature information in out directory name
        out_path = Path(
            f"{dataset}_{args['model_type']}_{dataset_type}_{feature_set}_{time.time()}"
        )
    out_path.mkdir(exist_ok=True)

    study_name = args["study_name"] or (
        f"{dataset}_search_{args['model_type']}_"
        f"{np.random.random_integers(0, 1000)}"
    )
    relative_n_trials = N_TRIALS - args["num_trials_completed"]
    logging.info(f"Setting up study {study_name} with {relative_n_trials} trials")
    study = load_study(
        study_name,
        out_path,
        # sampler=create_skopt_sampler,
        study_db_name="hyperparameter_search_fix.db",
    )
    logging.info(f"Study with sampler: {study.sampler}")
    sampler_cb = partial(save_sampler_cb, path=out_path)
    study.optimize(
        func=lambda trial: objective(
            trial,
            base_path,
            out_path,
            SEED,
            schema,
            model_type=args["model_type"],
            continuous_projection=args["continuous_projection"],
            aggregated_projection=args["aggregated_projection"],
        ),
        n_trials=relative_n_trials,  # N_TRIALS - args["num_trials_completed"],
        gc_after_trial=True,
        callbacks=[
            sampler_cb,
            empty_cache_cb,
            lambda _, trial: clean_out_dir_cb(trial, out_path),
        ],
        catch=[MemoryError],
    )
    logging.info(
        f"Found best parameters during trial \n{study.best_trial} "
        + f"with NDCG@10 at {study.best_value}"
    )
    logging.info(f"Best params:\n{study.best_params}")
    logging.info("COMPLETE")
