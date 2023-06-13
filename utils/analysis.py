import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from utils import SavePredictionsCallback
from utils.constants import SEED


def get_next_item_metrics(results_dict: t.Dict[str, pd.DataFrame]):
    select_metrics = [
        col
        for col in list(results_dict.values())[0]["predict"].columns
        if "next-item" in col
    ]
    res_keys, res_df = zip(*sorted(results_dict.items(), key=lambda pair: pair[0]))
    prediction_res = pd.concat(res_df, axis=0, keys=res_keys)["predict"][select_metrics]
    prediction_res.columns = [col.split("/")[-1] for col in prediction_res.columns]
    prediction_res.index.set_names(["type", "seed"], inplace=True)
    return prediction_res


def calculate_metrics(metrics: t.Dict[str, t.Any], ks=[5, 10]):
    results = {
        metric_name: torch.cat(metric.metric_mean, axis=0).mean(axis=0)
        for metric_name, metric in metrics.items()
    }
    metric_results = defaultdict(dict)

    for metric_name, result in results.items():
        for k_idx, topk in enumerate(ks):
            metric_results[metric_name][topk] = result[k_idx].item()
    return dict(metric_results)


def save_eval(metrics: t.Dict, storage: SavePredictionsCallback, out_name: str):
    """Helper function to save predictions and metrics"""
    predictions_path = Path("predictions")
    pred_out_dir = predictions_path / out_name
    if not pred_out_dir.exists():
        Path.mkdir(pred_out_dir)
    storage.predictions_to_parquet(pred_out_dir / f"seed_{SEED}.parquet")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_parquet(f"results/{out_name}.parquet")
    metrics_df.to_json(f"results/{out_name}.json", indent=4)


def get_baseline_metrics(result_dict: t.Dict[str, pd.DataFrame]):
    metrics = {
        baseline: _get_baseline_metric(result_df)
        for baseline, result_df in result_dict.items()
    }
    res_keys, res_df = zip(*sorted(metrics.items(), key=lambda pair: pair[0]))
    prediction_res = pd.concat(res_df, axis=0, keys=res_keys)
    prediction_res.index.set_names(["type", "seed"], inplace=True)
    return prediction_res


def _get_baseline_metric(baseline_df: pd.DataFrame):
    """Convert a baseline eval structure to one returned by the others"""
    df = baseline_df.rename(columns={"map": "avg_precision"})
    column_stacked = df.unstack().to_frame().T
    column_stacked.columns = [
        "_".join(col).strip() for col in column_stacked.columns.values
    ]
    return column_stacked


def rename_col(col_name: str) -> str:
    tokens = col_name.lower().split("_")
    k = int(tokens[-1])
    if "precision" in tokens and "avg" in tokens:
        return f"MAP@{k}"
    elif "recall" in tokens:
        return f"R@{k}"
    elif "ndcg" in tokens:
        return f"NDCG@{k}"
    else:
        raise RuntimeWarning(f"Unknown col name mapping {col_name}")
        raise RuntimeWarning(f"Unknown col name mapping {col_name}")
