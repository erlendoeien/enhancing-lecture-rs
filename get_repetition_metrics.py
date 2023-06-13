import logging
import typing as t
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat

import pandas as pd
import scipy.stats as ss
import torch
from tqdm import tqdm

from utils.analysis import calculate_metrics
from utils.io import setup_args, setup_logging
from utils.t4r_analysis import get_metrics

SEED = 481424852
PREDICT_K = 20
BATCH_SIZE = 1024 * 4


class RepetitionDataset(torch.utils.data.Dataset):
    def __init__(self, user_history: pd.DataFrame, prediction_df: pd.DataFrame):
        """Initialization
        Labels  - pd.DataFrame, indexed by `user_id` and with column `item_id`"""
        self.user_history = user_history
        self.prediction_df = prediction_df.reset_index()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.prediction_df)

    def __getitem__(self, index):
        "Generates one sample of data"
        user_id, _, preds, pred_scores = self.prediction_df.iloc[index].values
        history = self.user_history.loc[user_id]
        preds_tensor = torch.tensor(preds)
        labels = torch.isin(preds_tensor, torch.tensor(list(history))).int()
        # Don't actually need to return preds, only scores and labels
        return torch.tensor(pred_scores), labels


def get_unique_repetition_history(df: pd.DataFrame) -> pd.Series:
    """Get the unique items which are repeated per user.
    Note: Will only return users with repetition behahiour"""
    df_reset = df.reset_index(level="user_id")
    all_dups = df_reset.duplicated(["user_id", "item_id"], keep=False)
    return df_reset[all_dups].groupby("user_id")["item_id"].agg(set)


def get_unique_history(df: pd.DataFrame) -> pd.Series:
    """Get the unique items a user has interacted with
    TODO: Check output of groupby"""
    return df.groupby("user_id")["item_id"].agg(set)


def get_unique_rep_frac(
    unique_rep_history: pd.Series, unique_history: pd.Series
) -> pd.Series:
    """Calculates the ratio between unique rewatched items and the total
    unique item history of the user.
    Will only be calculated for users with repeating behaviour
    Note: Assuming that the series have the same index"""
    return (unique_rep_history.str.len() / unique_history.str.len()).dropna()


# Repetition Ranking helpers (Calculate metrics is almost identical to Non-personalised
# baseline versions
def predict_repetitions(loader, ks=[5, 10]):
    metrics = get_metrics(ks)
    for scores, labels in loader:
        for _, metric in metrics.items():
            metric.update(scores, labels)
    return metrics


def prepare_ranking_eval(user_history: pd.Series, predictions: pd.DataFrame, **kwargs):
    generator = torch.Generator().manual_seed(SEED)
    ds = RepetitionDataset(user_history, predictions)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, generator=generator, **kwargs
    )
    return loader, generator


def get_repetition_ranking_metrics(
    user_history: pd.Series,
    predictions: pd.DataFrame,
    ks=[5, 10],
    **kwargs,
):
    """Calculates ranking metrics of repetitions, where the ground truth
    is each user's training history.

    Args:
        user_history (pd.Series): A user_id -> set(interacted_items)-indexed series
        predictions (pd.DataFrame): The raw prediction format of
        user_id-> (prediction_items, prediction_scores, labels)
        seed (int, optional): Seed to use for the generator. Defaults to SEED.
        ks (torch.tensor, optional): Cut-offs. Defaults to [5,10].

    Returns:
        pd.DataFrame: The metrics for each cut_off
    """
    loader, _ = prepare_ranking_eval(user_history, predictions, **kwargs)
    batch_metrics = predict_repetitions(loader, ks=ks)
    metrics = pd.DataFrame(calculate_metrics(batch_metrics, ks=ks))

    # Convert to (metric, k) hieriarchical columns
    return metrics.unstack().to_frame().T


def get_adaptive_rec_rep_frac(
    predictions: pd.Series, user_repetition_set: pd.Series, ks=[5, 10]
):
    """Calculates the number of recommendations at top K which the user has already
    interacted with  and divides it by the number of unique items in each user's
    history.
    predictions: pd.Series with decoded `user_id` as index and a
                list of recommended items
    user_history_set: pd.Series with decoded `user_id` as index and
                    `history`:set as the (unique) encoded item_ids in the user's
                    past interaction history
    returns: pd.DataFrame
                A dataframe consisting of the number of repetitions recommended for each
                k, as well as the historic repetition fraction
    """
    recs = []
    # Calculate overlap with user history for top k recommended items
    for k in ks:
        sliced_preds = pd.concat(
            [
                predictions.str[:k].transform(set).rename("preds"),
                user_repetition_set.rename("rep_history"),
            ],
            axis=1,
        )
        overlaps = sliced_preds.apply(
            lambda row: row["preds"].intersection(row["rep_history"]), axis=1
        )
        # The number of suggested already seen items divided by the user's unique
        # item history size
        overlap_lengths = overlaps.str.len().rename(k)
        # rec_rep_frac = overlap_lengths / user_history_set.str.len()
        # Divide overlap with k as it is more general than individual users
        rec_rep_frac = overlap_lengths / k
        recs.append(rec_rep_frac)
    return pd.concat(recs, keys=ks, axis=1).astype(float)


def get_new_kl_divergence_at_k(
    predictions: pd.Series,
    user_first_time_set: pd.Series,
    user_repetition_set: pd.Series,
    user_history_set: pd.Series,
    ks=[5, 10],
    base=2,
):
    pass
    KEYS = ["first_watch", "rewatch"]
    # Replacing explore set (only seen once), a few have only repetitions
    historic_distribution = pd.concat(
        get_historic_class_length(user_first_time_set, user_repetition_set),
        keys=KEYS,
        axis=1,
    )
    divergences = []
    for k in ks:
        recommendation_distribution = pd.concat(
            get_new_prediction_class_length_at_k(predictions, user_history_set, k),
            keys=KEYS,
            axis=1,
        )

        assert (historic_distribution.index == recommendation_distribution.index).all()
        kl_divergence = pd.Series(
            ss.entropy(
                recommendation_distribution, historic_distribution, base=base, axis=1
            ),
            index=recommendation_distribution.index,
        )
        divergences.append(kl_divergence)

    return pd.concat(divergences, keys=ks, axis=1).astype(float)


def get_new_prediction_class_length_at_k(
    predictions: pd.Series,  # = None,
    user_history_set: pd.Series,
    k: int,
) -> t.Tuple[pd.Series, pd.Series]:
    """Calculates the number of predictions for each class.
    The difference is that now the exploit category includes all
    items seen by the user, not just those which are repeated.

    Args:
        predictions (pd.Series): The predicted item ids, lists
        user_history_set (pd.Series): Each user's history, sets
        k (int): The cut_off to evaluate it on

    Returns:
        tuple: The frequencies of each explore and exploit class respectively
    """
    assert (
        predictions.index == user_history_set.index
    ).all(), "Prediction index doesn't match"

    predictions = predictions.copy().str[:k].map(set)
    explore = predictions - user_history_set
    df = pd.concat([predictions, user_history_set], axis=1)
    exploit = df.apply(lambda row: row[0].intersection(row[1]), axis=1)

    return explore.str.len(), exploit.str.len()


def get_kl_divergence_at_k(
    predictions: pd.Series,
    user_first_time_set: pd.Series,
    user_repetition_set: pd.Series,
    ks=[5, 10],
    base=2,
):
    KEYS = ["first_watch", "rewatch"]
    # Replacing all zeros with a small value
    historic_distribution = pd.concat(
        get_historic_class_length(user_first_time_set, user_repetition_set),
        keys=KEYS,
        axis=1,
    )
    divergences = []
    for k in ks:
        recommendation_distribution = pd.concat(
            get_prediction_class_length_at_k(
                predictions, user_first_time_set, user_repetition_set, k
            ),
            keys=KEYS,
            axis=1,
        )
        all_zero = (recommendation_distribution == 0).all(axis=1)
        # When not recommending any, say that it is equally recommended
        recommendation_distribution[all_zero] = 1

        assert (historic_distribution.index == recommendation_distribution.index).all()
        kl_divergence = pd.Series(
            ss.entropy(
                recommendation_distribution, historic_distribution, base=base, axis=1
            ),
            index=recommendation_distribution.index,
        )
        divergences.append(kl_divergence)

    return pd.concat(divergences, keys=ks, axis=1).astype(float)


def get_historic_class_length(
    user_first_time_set: pd.Series,
    user_repetition_set: pd.Series,
):
    return user_first_time_set.str.len(), user_repetition_set.str.len()


def get_prediction_class_length_at_k(
    predictions: pd.Series,
    user_first_time_set: pd.Series,
    user_repetition_set: pd.Series,
    k: int,
):
    # First time set and rewatch set are disjoint
    assert (
        user_first_time_set.index == user_repetition_set.index
    ).all(), "Indexes aren't equal"
    assert (
        predictions.index == user_repetition_set.index
    ).all(), "Prediction index doesn't match"
    predictions = predictions.copy().str[:k].map(set)
    first_watches = predictions - user_repetition_set
    df = pd.concat([predictions, user_first_time_set, user_repetition_set], axis=1)
    first_watches = df.apply(lambda row: row[0].intersection(row[1]), axis=1)

    # not_watched = (first_watches - user_first_time_set) - user_repetition_set
    rewatches = df.apply(lambda row: row[0].intersection(row[2]), axis=1)
    assert isinstance(first_watches, pd.Series), (
        "Rewatch predictions not pd.Series" f"but {type(first_watches)}"
    )
    assert isinstance(rewatches, pd.Series), (
        "Rewatch predictions not pd.Series" f"but {type(rewatches)}"
    )

    return first_watches.str.len(), rewatches.str.len()  # , not_watched.str.len()


def evaluate_repetitions(
    dataset: t.Literal["mooc", "ednet"],
    user_history_set: pd.Series,
    int2user_id: t.Dict[int, t.Union[int, str]],
    user_repetition_frac: pd.Series = None,
    user_repetition_set: pd.Series = None,
    user_first_time_set: pd.Series = None,
    models=["baseline", "xlnet", "bert", "gru"],
    ks=[5, 10],
):
    """
    For a given dataset, evaluate the model on some repetition metrics.
    The repetition metrics for each model, for each seed is then concatenated and returned.
    """
    predictions_path = Path("predictions")
    model_results = {}
    for model in tqdm(models):
        # get all model variations for eval
        model_paths = list(predictions_path.glob(f"{dataset}_{model}*"))
        for model_path in model_paths:
            model_key_tokens = model_path.stem.split("_")
            model_key = (
                model_key_tokens[2:-2]
                if "baseline" in model_path.stem
                else model_key_tokens[1:-2]
            )
            logging.info(f"\tGetting repetition results for {'_'.join(model_key)}")
            seed_paths = list(model_path.glob("*.parquet"))
            model_metrics = evaluate_seeds(
                seed_paths,
                user_history_set,
                int2user_id,
                user_repetition_frac=user_repetition_frac,
                user_repetition_set=user_repetition_set,
                user_first_time_set=user_first_time_set,
                ks=ks,
            )
            model_results["_".join(model_key)] = model_metrics

    model_keys, model_dfs = zip(*model_results.items())
    results_df = pd.concat(model_dfs, keys=model_keys)
    results_df.index.set_names(["model", "seed"], inplace=True)
    return results_df


def evaluate_seeds(
    pred_paths: t.List[Path],
    user_history_set: pd.Series,
    int2user_id: t.Dict[int, t.Union[int, str]],
    user_repetition_frac: pd.Series = None,
    user_repetition_set: pd.Series = None,
    user_first_time_set: pd.Series = None,
    ks=[5, 10],
):
    """
    For a given set of seeds, corresponding to the different predictions of a model,
    calculate the resulting repetition metrics. Which metric scenario is calculated,
    depends on whether a learner historic repetition-fraction is given or not as the
    scenarios are disjoint.
    """
    with Pool(len(pred_paths)) as pool:
        seed_results = pool.map(
            partial(
                evaluate_seed,
                user_history_set=user_history_set,
                int2user_id=int2user_id,
                user_repetition_frac=user_repetition_frac,
                user_repetition_set=user_repetition_set,
                user_first_time_set=user_first_time_set,
                ks=ks,
            ),
            pred_paths,
        )
    seed_keys, seed_dfs = zip(*seed_results)
    return pd.concat(seed_dfs, keys=seed_keys).droplevel(1)


def evaluate_seed(
    seed_path: Path,
    user_history_set: pd.Series,
    int2user_id: t.Dict[int, t.Union[int, str]],
    user_repetition_frac: pd.Series = None,
    user_repetition_set: pd.Series = None,
    user_first_time_set: pd.Series = None,
    ks=[5, 10],
):
    seed = int(seed_path.stem.split("_")[-1])
    predictions = pd.read_parquet(seed_path)
    # During testing - User id's are not shuffled -> Map back to the correct user
    predictions.index = predictions.index.map(int2user_id).rename("user_id")

    # Won't calculate both at the same time
    if user_repetition_frac is None:
        # Calculate repetition ranking metrics of model
        result_df = get_repetition_ranking_metrics(user_history_set, predictions, ks=ks)
    else:
        # Downsample predictions and user_history_list -
        predictions_subset = predictions[
            predictions.index.isin(user_repetition_frac.index)
        ]
        # # Likely already subsetted
        # subset_user_history = user_history_set[
        #     user_history_set.index.isin(user_repetition_frac.index)
        # ]
        assert predictions.shape[0] > predictions_subset.shape[0], (
            "The number of users are not reduced"
            f"expected: {predictions.shape[0]}, actual: {predictions_subset.shape[0]}"
        )

        rep_rec_frac = get_adaptive_rec_rep_frac(
            predictions_subset["pred_items"], user_history_set, ks=ks
        )
        rep_rec_frac_df = pd.concat(
            [rep_rec_frac, user_repetition_frac.rename("historic_rep_frac")], axis=1
        )

        # Get Rec2Rep ratio for each k -> Avoids divide-by-zero
        # Subtract 1 to "normalize" around 0
        recs2reps = rep_rec_frac_df[ks].transform(
            lambda col: col.div(rep_rec_frac_df["historic_rep_frac"], axis="index")
        )
        recs2reps_summary = recs2reps.describe()

        # ALOT OF INF
        kl_divergence = get_kl_divergence_at_k(
            predictions_subset["pred_items"],
            user_first_time_set,
            user_repetition_set,
            ks,
        )
        kl_new_divergence = get_new_kl_divergence_at_k(
            predictions_subset["pred_items"],
            user_first_time_set,
            user_repetition_set,
            user_history_set,
            ks,
        )
        # rmse = get_rmse(rep_rec_frac_df, ks=ks)
        # print(recs2reps_summary.loc["mean"].to_frame().T)
        # print("Num divergence inf")
        # print((kl_divergence == np.inf).sum())
        result_df = (
            pd.concat(
                [
                    kl_divergence.mean(axis=0).to_frame().T,
                    kl_new_divergence.mean(axis=0).to_frame().T,
                    recs2reps_summary.loc["mean"].to_frame().T.reset_index(drop=True),
                ],
                keys=["kl-divergence", "kl-new-divergence", "rec2rep"],
                axis=1,
            )
            # .unstack()
            # .to_frame()
            # .T
        )
        # print(result_df)

    # Stringify k_metrics
    result_df.columns = result_df.columns.map(lambda tup: (tup[0], str(tup[1])))
    return seed, result_df


def get_norm_ks(df_train: pd.DataFrame, ks=[5, 10]) -> t.List:
    df_lengths = df_train.groupby("user_id")["item_id"].size()
    return list(
        sorted(
            list(
                set(
                    [
                        *ks,
                        df_lengths.median().astype(int),
                        df_lengths.mean().astype(int),
                    ]
                )
            )
        )
    )


def load_user_interactions(path):
    return pd.read_parquet(path, columns=["item_id"]).droplevel(
        ["item_consecutive_id", "session_id"]
    )


if __name__ == "__main__":
    parser = setup_args()
    parser.add_argument("--r2r-only", action="store_true")
    args = vars(parser.parse_args())
    dataset, dataset_type, feature_set, log_name, out_dir = (
        args["dataset"],
        args["dataset_type"],
        args["feature_set"],
        args["log_file"],
        args["out_dir"],
    )
    log_dir = Path("logs")
    setup_logging(log_dir / log_name, level=logging.INFO)
    logging.info(f"Input arguments:\n{pformat(args, indent=4)}")
    base_path = Path(dataset) / dataset_type / feature_set

    out_base_path = Path(out_dir or "repetition_results")
    r2r_out = out_base_path / "rec2rep"
    r2r_out.mkdir(parents=True, exist_ok=True)
    ranking_out = out_base_path / "ranking_metrics"
    ranking_out.mkdir(parents=True, exist_ok=True)

    # Loading train dataset as this is what the models have learned on
    logging.info("Loading training sets")
    train = load_user_interactions(base_path / "train.parquet")

    logging.info("Calculating means and medians for cut-offs")
    ks = get_norm_ks(train)

    logging.info("Getting unique user histories")
    unique_user_history = get_unique_history(train)

    logging.info("Getting unique repetition histories")
    unique_user_repetition_history = get_unique_repetition_history(train)

    logging.info("Calculating unique historic repetition fraction")
    unique_repetition_frac = get_unique_rep_frac(
        unique_user_repetition_history, unique_user_history
    )
    assert (unique_repetition_frac <= 1).all(), "Invalid repetition fraction"

    logging.info("Loading test sets for explicit user2prediction mapping")
    test = load_user_interactions(base_path / "test.parquet")

    int2user_id = {num: user_id for num, user_id in enumerate(test.index)}

    # Already completed # RQ3a
    # if not RANKING_METRICS_DONE:
    if not args["r2r_only"]:
        logging.info(f"Calculating Ranking metrics for {dataset}")
        repetition_ranking_metrics = evaluate_repetitions(
            dataset,
            unique_user_history,
            int2user_id,
            ks=ks,
        )

        logging.info(f"Storing ednet ranking metrics to {ranking_out}")
        repetition_ranking_metrics.to_parquet(
            ranking_out / f"{dataset}_metrics.parquet"
        )
    # RQ3b
    # Have repeated at least once ->
    # user_history_set.length < total_user_interaction.length
    # Get user_ids which satisfy it
    logging.info(
        f"Num users with less than 1 rep frac: {(unique_repetition_frac < 1).sum()}"
    )
    logging.info(f"Num users with 1 rep frac: {(unique_repetition_frac == 1).sum()}")
    logging.info(
        f"Num users with repetition frac > 0: {(unique_repetition_frac > 0).sum()}"
    )
    logging.info(f"Num users with repetition frac: {unique_repetition_frac.shape[0]}")

    # Only include Users with both repeating and single watch behaviour
    repeating_frac_both = unique_repetition_frac[unique_repetition_frac < 1]
    # Filter predictions, user_history and test_set on this
    repeating_users_history = unique_user_history.loc[repeating_frac_both.index]
    # Items which the user has not rewatched
    repeating_users_repetition_history = unique_user_repetition_history[
        repeating_frac_both.index
    ]

    user_first_time_set = repeating_users_history - repeating_users_repetition_history
    # First time set and rewatch set are disjoint
    assert (
        user_first_time_set.index == repeating_users_repetition_history.index
    ).all(), "Indexes aren't equal"
    assert (
        user_first_time_set.index == repeating_frac_both.index
    ).all(), "Indexes aren't equal"

    logging.info("Excluding users with only repetition behaviour")
    logging.info(
        f"Num users without first_watches: {(user_first_time_set.str.len() == 0).sum()}"
    )
    logging.info(
        "Num users without rewatches: "
        f"{(unique_user_repetition_history.str.len() == 0).sum()}"
    )
    logging.info(
        f"Num users with repeat history (and first time): {repeating_frac_both.shape[0]}"
    )

    logging.info(ks)
    logging.info(f"Calculating R2R metrics for {dataset}")
    repeating_users_r2r = evaluate_repetitions(
        dataset,
        repeating_users_history,
        int2user_id,
        user_repetition_frac=repeating_frac_both,  # unique_repetition_frac,
        user_repetition_set=repeating_users_repetition_history,
        user_first_time_set=user_first_time_set,
        ks=ks,
    )

    logging.info(f"Storing ednet rec2rep metrics to {r2r_out}")
    repeating_users_r2r.to_parquet(r2r_out / f"{dataset}_metrics.parquet")
    logging.info("COMPLETE")
