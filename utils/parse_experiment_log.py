import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def get_metrics(log_lines: list, end_of_metrics_keywords: str, num_metric_lines=10):
    end_of_pred_round_lines = [
        (num, line) for num, line in log_lines if end_of_metrics_keywords in line
    ]
    metrics_raw = [(end - num_metric_lines, end) for end, _ in end_of_pred_round_lines]
    metric_results = []
    for start, end in metrics_raw:
        metric_str = " ".join(
            [line.replace("'", '"') for _, line in log_lines[start:end]]
        )
        metrics = json.loads(metric_str)
        metric_results.append(metrics)
    return metric_results


def get_predictions_stat(operator, metrics: list):
    """Get the statistic across multiple runs for each metric"""
    return {
        key: operator([metrics[key] for metrics in metrics])
        for key in metrics[0].keys()
    }


def print_metrics(means: dict, stds: dict):
    for key in means.keys():
        print(f"\t{key}: {means[key]:.4f} +- {stds[key]:.4f}")


def result_index2seed(
    log_lines: list,
    indices: list,
):
    regex = (
        r"Training ["
        + "|".join([str(idx) for idx in indices])
        + r"] with seed (\d{1,4})"
    )
    seeds_str = re.findall(regex, "\n".join([line for _, line in log_lines]))
    # Remove duplicates, keep first, insertion ordered for Python +3.6
    return list(dict.fromkeys([int(seed) for seed in seeds_str]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for parsing the evaluation metrics"
        "from the logs of an eval script"
    )
    parser.add_argument(
        "--experiment-log",
        "-e",
        type=str,
        required=True,
        help="Path to experiment log file to analyse, created from `evaluate_model.py`",
    )
    parser.add_argument(
        "-out-dir",
        "-o",
        type=str,
        help="Path to out directory to save results",
    )
    args = vars(parser.parse_args())
    experiment_log_path = Path(args["experiment_log"])
    with open(experiment_log_path) as f:
        log_lines = list(enumerate(f.read().splitlines()))
    num_metric_lines = 10
    prediction_results = get_metrics(log_lines, "Saving prediction data to")
    # evaluation_results = get_metrics(log_lines, "PREDICT")

    # eval_means = get_predictions_stat(np.mean, evaluation_results)
    # eval_stds = get_predictions_stat(np.std, evaluation_results)

    prediction_means = get_predictions_stat(np.mean, prediction_results)
    prediction_stds = get_predictions_stat(np.std, prediction_results)

    # print("Evaluation results")
    # print_metrics(eval_means, eval_stds)
    # print("\n" + "-" * 10 + "\n")
    # print("Prediction results")
    print_metrics(prediction_means, prediction_stds)

    if (out_dir := args["out_dir"]) is not None:
        out_name = experiment_log_path
        indices = list(range(len(prediction_results)))
        idx2seed = dict(zip(indices, result_index2seed(log_lines, indices)))

        # results = {}
        # eval_seeds = {
        #     idx2seed[idx]: eval_res for idx, eval_res in enumerate(evaluation_results)
        # }
        results = {
            idx2seed[idx]: pred_res for idx, pred_res in enumerate(prediction_results)
        }
        # for seed in [idx2seed[idx] for idx in indices]:
        #     break
        # seed_results = {
        #     "eval": eval_seeds[seed],
        #     "predict": pred_seeds[seed],
        # }
        # results[seed] = {
        #     (val_type, metric): value
        #     for val_type, metrics in seed_results.items()
        #     for metric, value in metrics.items()
        # }
        out_path = Path(out_dir) / experiment_log_path.stem
        print(f"Saving metrics to {out_path}")
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.to_parquet(f"{out_path}.parquet")
        results_df.to_json(f"{out_path}.json", indent=4, orient="index")
