import json
import logging
from pathlib import Path

from evaluate_model import CAT_COLS, FLOAT_COLS
from utils.io import setup_logging

DATASETS = ["ednet", "mooc"]
# Including only 1 segment repetition feature
SEG_REP_FEAT = "seg_rep_60"

if __name__ == "__main__":
    setup_logging("logs/create_feature_configs.log")
    config_path = Path("configs")

    for dataset in DATASETS:
        ALL_FLOATS = FLOAT_COLS(dataset)
        selected_floats = [
            feat for feat in ALL_FLOATS if not feat.startswith("seg_rep")
        ] + [SEG_REP_FEAT]

        dataset_config = {
            col: [False, True] for col in selected_floats + CAT_COLS(dataset)
        }
        logging.info(f"Created powerset feature config for {dataset}:\n")
        out_path = config_path / f"{dataset}_features_config.json"
        logging.info(f"Storing feature config to {out_path}")
        with open(out_path, "w") as f:
            json.dump(dataset_config, f, indent=4)
