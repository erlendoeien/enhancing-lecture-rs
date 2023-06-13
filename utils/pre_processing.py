# Feature Pre-processing
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocesspandas import applyparallel  # noqa

from utils.constants import MAX_SEQUENCE_LENGTH, TIME_COL, USER_COL

CPU_COUNT = 1  # multiprocessing.cpu_count()
# DEFAULTS


def norm_vid_feats(
    df: pd.DataFrame, max_lengths: pd.DataFrame, item_col="item_id"
) -> pd.DataFrame:
    """Normalize video specific features like `time_completed` by the maximum recorded
    value for that given feature for each item. For OOV, fill in 1 if value is present
    in test, but not in training and if both test and train encounters are 0.
    `max_lengths` is a dataframe with the max of the features for each
    item to normalize"""
    df = df.copy()

    df_group = df.groupby(item_col, group_keys=False, sort=False)[max_lengths.columns]
    if CPU_COUNT == 1:
        df_normed = df_group.progress_apply(
            lambda group: group / max_lengths.loc[group.name]
        )
    else:
        df_normed = df_group.apply_parallel(
            lambda group: group / max_lengths.loc[group.name]
        )

    df[max_lengths.columns] = df_normed.replace(np.inf, 1).fillna(0)

    return df


def listify(df: pd.DataFrame, user_col=USER_COL, time_col=TIME_COL) -> pd.DataFrame:
    """Aggregates all columns into lists for each user."""
    return (
        df.reset_index()
        .sort_values(time_col)
        .groupby(user_col, group_keys=False, sort=False)
        .agg(list)
    )


def pad(series: pd.Series, pad_token=0, max_length=MAX_SEQUENCE_LENGTH) -> pd.Series:
    """Pads a given Series which consists of sliced lists of `max_length`"""
    return series.transform(lambda lst: (lst + [pad_token] * (max_length - len(lst))))


def pad_split(
    df: pd.DataFrame,
    reg_cols=None,
    list_cols=None,
    pad_token=0,
    max_list_length=None,
    **kwargs,
) -> pd.DataFrame:
    """Pad a given listified dataset split with the supplied token.
    With multi-label columns of the shape (SEQ_LEN, 1, LABEL_LEN)),
    use the supplied list length for padding"""
    df = df.copy()
    if reg_cols is not None:
        df.loc[:, reg_cols] = df[reg_cols].transform(
            pad, pad_token=pad_token, **kwargs, axis=0
        )
    if list_cols is not None:
        df.loc[:, list_cols] = df[list_cols].transform(
            pad, pad_token=[[pad_token] * max_list_length], **kwargs, axis=0
        )
    return df


def bulk_listify(*dfs, **kwargs):
    return [listify(df, **kwargs) for df in dfs]


def bulk_pad_split(*dfs, reg_cols, list_cols=None, **kwargs):
    return [pad_split(df, reg_cols, list_cols, **kwargs) for df in dfs]


def save_splits(dir_path: Path, split_dict: dict):
    for split_name, split in split_dict.items():
        split.to_parquet(dir_path / f"{split_name}.parquet")


def minmax_norm(series: pd.Series) -> pd.Series:
    """Does minmax-normalization, currently unused"""
    if not series.any():
        return series
    return ((series - series.min()) / (series.max() - series.min())).fillna(0)


def adaptive_z_score(series: pd.Series) -> pd.Series:
    """Adaptively standardizes the segment repetition as one learns more about
    the user's repetition behaviour. Currently unused"""
    if not series.any():
        return series
    expanding_window = series.expanding(1)
    return ((series - expanding_window.mean()) / expanding_window.std()).fillna(0)


def adaptive_bias(series: pd.Series) -> pd.Series:
    """Calculating the adaotuve bias adjusted score.
    Works with time based LOO because it will be adjusted by the existing known mean of
    the user. ALTERNATIVE: Make it a rolling bias"""
    if not series.any():
        return series
    return series - series.expanding(1).mean()
