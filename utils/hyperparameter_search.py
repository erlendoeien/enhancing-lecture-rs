import logging
import pickle
import typing as t
from pathlib import Path

import optuna

from utils.constants import SEED
from utils.io import load_json


def clean_out_dir_cb(trial: optuna.trial.FrozenTrial, out_path: Path):
    """Clean up trials to avoid data overload"""
    trial_path = out_path / f"trial_{trial.number}"
    logging.info(f"Cleaning up {trial_path}")
    if trial_path.exists() and trial_path.is_dir():
        for chckpt_dir in trial_path.iterdir():
            if chckpt_dir.is_dir():
                for file in chckpt_dir.iterdir():
                    if file.stem != "trainer_state":
                        file.unlink()


def save_sampler_cb(study, _, path=Path(".")):
    """For saving a Optuna Sampler state"""
    with open(path / "sampler.pickle", "wb") as fout:
        pickle.dump(study.sampler, fout)


# Inspired by tuning code from Daisy v2
def config_to_trial_param(config: dict, trial: optuna.Trial):
    trial_params = {}
    for param, value in config.items():
        if isinstance(value, list) and len(value) > 1:
            trial_params[param] = trial.suggest_categorical(param, value)
        elif isinstance(value, dict):
            if value.get("log") is not None:
                chunk_method = {"log": True}
            elif step := value["step"]:
                chunk_method = {"step": step}
            else:
                chunk_method = {"step": 1}

            if value["type_"] == "int":
                trial_params[param] = trial.suggest_int(
                    param, value["min"], value["max"], **chunk_method
                )
            elif value["type_"] == "float":
                trial_params[param] = trial.suggest_float(
                    param, value["min"], value["max"], **chunk_method
                )
            else:
                raise ValueError(f"Invalid parameter type for {param}...")
        else:
            # Constant param
            trial_params[param] = value
    return trial_params


def parse_hyperparams(hyper_parameters: dict, is_item_only=False):
    """Parses the optimal hyperparameters to the specific module parameters,
    overriding the hyper parameter configurations and maintaining the static
    configurations"""
    configs_path = Path("configs")
    input_config = load_json(configs_path / "config_input.json")
    model_config = load_json(configs_path / "config_model.json")
    training_args_config = load_json(configs_path / "config_trainer_args.json")
    input_args = {
        **input_config,
        **{
            key: value for key, value in hyper_parameters.items() if key in input_config
        },
    }
    if not is_item_only:
        side_info_input_config = load_json(configs_path / "config_side_info_input.json")
        input_args = {
            **input_args,
            **side_info_input_config,
            **{
                key: value
                for key, value in hyper_parameters.items()
                if key in side_info_input_config
            },
        }
    model_args = {
        **model_config,
        **{
            key: value for key, value in hyper_parameters.items() if key in model_config
        },
    }
    training_args = {
        **training_args_config,
        **{
            key: value
            for key, value in hyper_parameters.items()
            if key in training_args_config
        },
    }
    return input_args, model_args, training_args


def load_study(
    study_name: str,
    base_path: Path,
    # renew_sampler=False,
    sampler=optuna.samplers.TPESampler,
    search_space: t.Dict = None,
    study_db_name=None,
) -> optuna.Study:
    """Loads the given study from the default database.
    It will also load any existing `sampler.pickle` in the `base_path`"""
    study_db_name = study_db_name or "base_xlnet_search.db"
    logging.info(f"Looking up studies in {study_db_name}")
    study_db = f"sqlite:///{study_db_name}"

    storage = optuna.storages.RDBStorage(
        url=study_db,
        heartbeat_interval=90,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )

    sampler_path = base_path / "sampler.pickle"
    if sampler_path.exists():  # and not renew_sampler:
        logging.info("Loading existing sampler")
        sampler = pickle.load(open(sampler_path, "rb"))
    else:
        logging.info("Creating a new sampler")
        if sampler == optuna.samplers.GridSampler:
            sampler = sampler(search_space, seed=SEED)
        else:
            sampler = sampler(seed=SEED)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    return study


# def create_skopt_sampler(seed: int):
#     """Wrapper to create Skopt Sampler with RandomSample as independent sampler"""
#     # Using the default base estimator Gaussian Process and default
#     # 10 initial points /random starts
#     skopt_kwargs = {"random_state": seed, "base_estimator": "GP"}
#     return optuna.integration.SkoptSampler(seed=seed, skopt_kwargs=skopt_kwargs)
