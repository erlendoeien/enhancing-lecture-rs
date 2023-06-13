import json
from pathlib import Path

import numpy as np
import pandas as pd


class SavePredictionsCallback:
    """Sudo-Callback for accumulating predictions for each `prediction_step`"""

    def __init__(self) -> None:
        self.labels = []
        self.pred_items = []
        self.pred_scores = []
        self.collection: dict = None

    def __call__(self, **kwargs):
        # Save the current prediction step
        self.labels.append(kwargs["labels"])
        self.pred_items.append(kwargs["pred_item_ids"])
        self.pred_scores.append(kwargs["pred_item_scores"])

    def collect(self):
        self.collection = {
            "labels": np.hstack(self.labels).tolist(),
            "pred_items": np.vstack(self.pred_items).tolist(),
            "pred_scores": np.vstack(self.pred_scores).tolist(),
        }

    def predictions_to_parquet(self, path: Path, recollect=False):
        # with open(str(prediction_path) + ".json", "w") as f:
        #     json.dump(prediction_data, f, indent=4)
        if self.collection is None or recollect:
            self.collect()
        df = pd.DataFrame(self.collection)
        df.to_parquet(path)
        return df

    def predictions_to_json(self, path, recollect=False):
        if self.collection is None or recollect:
            self.collect()
        with open(path, "w") as f:
            json.dump(path, f, indent=4)
