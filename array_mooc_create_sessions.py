import logging
import sys
from pathlib import Path

import pandas as pd

from utils.io import setup_logging

ITEM_COL = "video_id"
USER_COL = "user_id"
TIME_COL = "local_start_time"
END_TIME_COL = "local_end_time"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"
FW_GAP_COL = "forward_gap"
BACK_GAP_COL = "backward_gap"
GAP_THRESH = pd.Timedelta("00:20:00")
BLACK_LIST_THRESH = 50


def flatten_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in a raw user-video dataframe and flattens all the segments and sequences
    for each user. Each user can have multiple records for the same video_id,
    but these are not differentiated in the flattening."""
    sequences = raw_df["seq"].explode()
    # Merge in each user's video_id entry
    sequences_merged = pd.concat([raw_df[USER_COL], sequences], axis=1).reset_index(
        drop=True
    )
    sequences_merged[ITEM_COL] = sequences_merged["seq"].str[ITEM_COL]
    # Merge together all recorded segments
    segments_merged = pd.concat(
        [
            sequences_merged.drop(columns="seq"),
            sequences_merged["seq"].str["segment"].explode().rename("segment"),
        ],
        axis=1,
    )
    segment_cols = ["start_point", "end_point", "speed", TIME_COL]
    segments_merged[segment_cols] = pd.DataFrame(
        {col: segments_merged["segment"].str[col] for col in segment_cols}
    )
    return segments_merged.drop(columns="segment").sort_values(TIME_COL)


def calculate_consecutive_user_item_ids(
    interactions_df,
    user_col=USER_COL,
    item_col=ITEM_COL,
    consecutive_id=CONSECUTIVE_COL,
):
    """Calculates the consecutive item ids for each user to
    generate the global interaction session ids. Assumes an already time-sorted DF"""
    interactions_df = interactions_df.reset_index(drop=True)
    return interactions_df.assign(
        **{
            consecutive_id: interactions_df.groupby(user_col, group_keys=False).apply(
                lambda group: (
                    group[item_col].shift() != group[item_col]
                ).cumsum(),  # meta=(item_col, "int64")
            )
        }
    )


def calculate_segment_gap(segments_df: pd.DataFrame):
    """For each row, calculate the difference between the current row and the next.
    The end time stamp is calculated using true watch time, accounting for PBR."""
    # logging.info("Calculating the segment gap for each user and consecutive video id")
    segments_df[END_TIME_COL] = segments_df[TIME_COL] + (
        (segments_df["end_point"] - segments_df["start_point"]) / segments_df["speed"]
    )
    segments_grouped = segments_df.groupby(
        [USER_COL, CONSECUTIVE_COL], group_keys=False
    )
    return segments_df.assign(
        **{
            FW_GAP_COL: segments_grouped[TIME_COL].shift(-1)
            - segments_df[END_TIME_COL],
            BACK_GAP_COL: segments_df[TIME_COL]
            - segments_grouped[END_TIME_COL].shift(),
        }
    )


def generate_sessions(consecutive_df, gap_thresh=GAP_THRESH, gap_col=BACK_GAP_COL):
    """Generate sessions based on the gap between same consecutive item interaction"""
    consecutive_df.loc[:, f"{BACK_GAP_COL}_td"] = pd.to_timedelta(
        consecutive_df[gap_col], unit="s"
    ).round("1s")
    consecutive_group = consecutive_df.groupby(
        [USER_COL, CONSECUTIVE_COL], group_keys=False
    )
    return consecutive_df.assign(
        **{
            SESSION_COL: consecutive_group.apply(
                lambda group: (group[f"{gap_col}_td"] > gap_thresh).cumsum()
            )
        }
    )


def remove_blacklist_users(session_df: pd.DataFrame, repeat_thresh=BLACK_LIST_THRESH):
    """Remove users who have seen one CCID determined video more than thresh times"""
    user_ccid_session_count = session_df.groupby([USER_COL, "ccid", CONSECUTIVE_COL])[
        SESSION_COL
    ].nunique()
    # Calculate num sessions per ccid per user and take the maximum of CCIDs count per user
    user_ccid_max_rep = (
        user_ccid_session_count.groupby([USER_COL, "ccid"])
        .sum()
        .groupby(USER_COL)
        .max()
        .sort_values(ascending=False)
    )

    blacklist = user_ccid_max_rep[user_ccid_max_rep > repeat_thresh]
    return session_df[~(session_df[USER_COL].isin(blacklist.index))], blacklist


if __name__ == "__main__":
    partition_num = int(sys.argv[1])
    setup_logging(
        Path("./logs") / "mooc_create_sessions_fix" / f"part.{partition_num}.log"
    )

    base_path = Path("~/fall_project/MOOCCubeX/")
    relations_path = base_path / "relations"
    video_id2ccid_path = "video_id-ccid.txt"
    partition_path = relations_path / "user2video_partitions"
    sessions_out_path = relations_path / "sessions_with_context_fix"
    sessions_out_path.mkdir(exist_ok=True, parents=True)

    in_out_name = f"part.{partition_num}.parquet"

    logging.info(f"Loading partition {in_out_name}")
    raw_df = pd.read_parquet(partition_path / in_out_name).reset_index()

    logging.info("Flattening df to records")
    interactions_df = flatten_df(raw_df)

    logging.info("Calculating the consecutive item ids for each user")
    consecutive_df = calculate_consecutive_user_item_ids(interactions_df)

    logging.info("Calculating segment gaps")
    gaps_df = calculate_segment_gap(consecutive_df)

    logging.info("Generating session ids per consecutive id")
    session_df = generate_sessions(gaps_df)
    logging.info(
        f"#interactions: {session_df.shape[0]}, #users: {session_df[USER_COL].nunique()}"
        f", #items: {session_df[ITEM_COL].nunique()}"
    )
    logging.info(
        f"#sessions: {session_df.groupby([USER_COL, CONSECUTIVE_COL])[SESSION_COL].nunique().sum()}"
    )

    logging.info("Starting blacklist process")
    logging.info("\tLoading video - ccid mapping")
    video_id2ccid = pd.read_csv(
        relations_path / "video_id-ccid.txt", sep="\t", names=[ITEM_COL, "ccid"]
    )
    logging.info("\tAdd in CCID")
    sessions_ccid = video_id2ccid.merge(session_df, how="right", on=ITEM_COL)

    logging.info("\tGenerating blacklist and cleaning dataset")
    sessions_clean, blacklist = remove_blacklist_users(sessions_ccid)

    logging.info(
        f"\tLocal number of blacklisted users: {blacklist.shape[0]}"
        f" out of {sessions_ccid[USER_COL].nunique()}"
    )
    n_records, cleaned_n_records = sessions_ccid.shape[0], sessions_clean.shape[0]
    logging.info(
        f"\tNumber of records reduced from {n_records} to {cleaned_n_records}"
        f" total {n_records - cleaned_n_records:,} removed"
        f"({(n_records - cleaned_n_records) / n_records:.4f})"
    )
    logging.info("Post cleaning")
    logging.info(
        f"#interactions: {sessions_clean.shape[0]}, #users: {sessions_clean[USER_COL].nunique()}"
        f", #items: {sessions_clean[ITEM_COL].nunique()}"
    )
    logging.info(
        f"#sessions: {sessions_clean.groupby([USER_COL, CONSECUTIVE_COL])[SESSION_COL].nunique().sum()}"
    )

    logging.info("Saving blacklist")
    blacklist_out = Path("pre_processing") / "mooc_blacklists_fix"
    blacklist_out.mkdir(exist_ok=True, parents=True)
    blacklist.to_csv(blacklist_out / f"{'.'.join(in_out_name.split('.')[:-1])}.csv")

    logging.info("Adding video context")
    logging.info("\tLoading video context")
    video_df = pd.read_parquet(base_path / "entities" / "video_lengths.parquet")
    logging.info("\tMerging together video context with sessions by ccid")
    sessions_context = video_df.merge(
        sessions_clean, left_index=True, right_on="ccid", how="right"
    ).sort_values(TIME_COL)
    logging.info("Cleaning and adding segment duration")
    sessions_context["concept_id"] = (
        sessions_context["concept_id"].fillna("").apply(list)
    )
    sessions_context["duration"] = (
        sessions_context["end_point"] - sessions_context["start_point"]
    )
    logging.info(
        f"Number of negative intervals: {(sessions_context['duration'] < 0).sum()}"
    )

    logging.info(
        f"Storing cleaned and sessionified dataset to {sessions_out_path / in_out_name}"
    )
    sessions_context.to_parquet(sessions_out_path / in_out_name)
    logging.info("COMPLETE")
