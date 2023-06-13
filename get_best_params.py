import argparse
import json
from pathlib import Path

import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for loading and storing the optimal"
        "hyperparameters as json, based on the given studies"
    )
    parser.add_argument(
        "--study-db", "-d", type=str, help="Relative path of the SQLite database"
    )
    parser.add_argument(
        "--study-names",
        "-n",
        nargs="+",
        help="The name of the studies to retrieve the best trial parameters from",
    )
    parser.add_argument(
        "-out-dir",
        "-o",
        type=str,
        help="Path to out directory for optimal hyperparameters for each study",
    )
    args = vars(parser.parse_args())
    study_db_path, studies, out_dir = (
        args["study_db"],
        args["study_names"],
        args["out_dir"],
    )
    out_path = Path(out_dir or "hyperparameters")
    if not out_path.exists():
        out_path.mkdir()
    study_db = f"sqlite:///{study_db_path}"

    storage = optuna.storages.RDBStorage(
        url=study_db,
        heartbeat_interval=90,
    )
    for study_name in studies:
        print(f"Loading best trial parameters from study {study_name}")
        study = optuna.study.load_study(study_name=study_name, storage=study_db)
        best_params = study.best_params
        out_json = out_path / f"{study_name}.json"
        print("\tSaving hyperparameters to", out_json)
        with open(out_json, "w") as f:
            json.dump(best_params, f, indent=4)
