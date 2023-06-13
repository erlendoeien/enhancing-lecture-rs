import logging
import sys
from pathlib import Path

import pandas as pd

from utils.io import setup_logging

ITEM_COL = "item_id"
USER_COL = "user_id"
TIME_COL = "timestamp"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"
BACK_GAP_COL = "back_gap"
FW_GAP_COL = "forward_gap"
# DUP_COLS = ["action_type", "cursor_time"]
# DUP_THRESH = pd.Timedelta("00:00:01")
PLAY_TYPE = "play_video"
PAUSE_TYPE = "pause_video"
ENTER_TYPE = "enter"
QUIT_TYPE = "quit"
GAP_THRESH = pd.Timedelta("00:20:00")
BLACK_LIST_THRESH = 50


def calculate_consecutive_user_item_ids(
    interactions_df,
    time_col=TIME_COL,
    user_col=USER_COL,
    item_col=ITEM_COL,
    consecutive_id=CONSECUTIVE_COL,
):
    """Calculates the consecutive item ids for each user to
    generate the global interaction session ids"""
    # FIX CUMSUM META
    return interactions_df.assign(
        **{
            consecutive_id: interactions_df.groupby(user_col, group_keys=False).apply(
                lambda group: (
                    group[item_col].shift() != group[item_col]
                ).cumsum(),  # meta=(item_col, "int64")
            )
        }
    )


def calculate_action_gaps(consecutive_df, back_gap=BACK_GAP_COL, fw_gap=FW_GAP_COL):
    """Calculate gaps between actions (forward and backwards) to use for session generation and feature extraction"""
    grouped = consecutive_df.groupby([USER_COL, CONSECUTIVE_COL], group_keys=False)
    return consecutive_df.assign(
        **{
            back_gap: (
                grouped[TIME_COL]
                .apply(lambda group: group.diff())
                .fillna(0)
                .astype("timedelta64[ms]")
            ),
            fw_gap: (
                grouped[TIME_COL]
                .apply(lambda group: group.shift(-1) - group)
                .fillna(0)
                .astype("timedelta64[ms]")
            ),
        }
    )


def generate_sessions(
    group, gap_thresh=GAP_THRESH, action_col="action_type", dur_col=BACK_GAP_COL
):
    """Create a new session if the play duration is largen than twice the video length
    or if the gap is larger than a set threshold"""
    # If previous was play and back_gap is large than 2 times video length -> Split
    # If previous not play -> check normal gap threshold
    is_prev_play = group[action_col].shift() == PLAY_TYPE
    # 25 mins is roughly the maximum video length, so to account for watching the entire video + the gap thresh at speed 1
    # In the cases where the video estimates are completely wrong
    max_play_gap = max(
        gap_thresh + pd.Timedelta("00:25:00"),
        pd.Timedelta(group["length"].iat[0] * 2, unit="ms"),
    )
    is_large_dur = group[dur_col] > max_play_gap
    is_play_gap = is_prev_play & is_large_dur
    is_other_gap = group[dur_col] > gap_thresh
    return (is_play_gap | ((~is_prev_play) & (is_other_gap))).cumsum()


# DEPRECATED
def old_generate_sessions(
    user_item_group,
    session_col=SESSION_COL,
    thresh=pd.Timedelta("00:10:00"),
    action_col="action_type",
):
    """Only for splitting up based on time, can remove recurring same-cursor actions later
    TODO: Improve, super slow"""
    counter = 0
    prev_type = ""
    for idx, row in user_item_group.iterrows():
        # Ignore re-entering and quiting the video if it is the consecutive id and within the threshold
        # New session if the pause duration is too high or the user has exited and re-joined the video within the last x time
        has_paused = row[action_col] == PLAY_TYPE and prev_type == PAUSE_TYPE
        has_re_entered = row[action_col] == ENTER_TYPE and prev_type == QUIT_TYPE
        if (has_paused or has_re_entered) and (row[BACK_GAP_COL] > thresh):
            counter += 1
        user_item_group.loc[idx, SESSION_COL] = counter
        prev_type = row[action_col]
    return user_item_group


# DEPRECATED - Only counting play->pause intervals anywho
# def filter_consecutive_actions(
#     df,
#     group_by_cols=[USER_COL, CONSECUTIVE_COL, SESSION_COL],
#     dup_cols=DUP_COLS,
#     dur_col=BACK_GAP_COL,
# ):
#     """Removing the the first action (I believe) of consecutive actions, similar to drop_duplicates("first")
#     Based on a threshold ("""
#     return df.groupby(group_by_cols, group_keys=False).apply(
#         lambda group: (group[dup_cols].shift() != group[dup_cols]).any(axis=1)
#         | (group[dur_col] > DUP_THRESH)
#     )


def remove_blacklist_users(session_df: pd.DataFrame, repeat_thresh=BLACK_LIST_THRESH):
    """Remove users who have seen one CCID determined video more than thresh times"""
    user_ccid_session_count = session_df.groupby([USER_COL, ITEM_COL, CONSECUTIVE_COL])[
        SESSION_COL
    ].nunique()
    # Calculate num sessions per ccid per user and take the maximum of CCIDs count per user
    user_item_max_rep = (
        user_ccid_session_count.groupby([USER_COL, ITEM_COL])
        .sum()
        .groupby(USER_COL)
        .max()
        .sort_values(ascending=False)
    )

    blacklist = user_item_max_rep[user_item_max_rep > repeat_thresh]
    return session_df[~(session_df[USER_COL].isin(blacklist.index))], blacklist


def clean_sessions(group):
    """Removing sessions if they only have 1 interaction and the user only has one session for that interaction.
    If not -> Could indicate wrong video and should exit"""
    if group[SESSION_COL].nunique() == 1:
        return group

    group = group.groupby(SESSION_COL).filter(lambda session: len(session) > 1)
    # Re-encode session-ids to still be consecutive
    group[SESSION_COL] = (group[SESSION_COL] != group[SESSION_COL].shift()).cumsum()
    return group


if __name__ == "__main__":
    partition_num = int(sys.argv[1])
    logs_path_dir = Path("./logs") / "ednet_create_sessions_fix"
    logs_path_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(logs_path_dir / f"part.{partition_num}.log")

    ednet_path = Path("~/fall_project/EdNet")
    lectures_path = ednet_path / "KT4_lectures"
    sessions_out_path = ednet_path / "KT4_sessions_fix"
    sessions_out_path.mkdir(exist_ok=True, parents=True)

    in_out_name = f"part.{partition_num}.parquet"

    logging.info(f"Loading partition {in_out_name}")
    lectures_df = (
        pd.read_parquet(lectures_path / in_out_name).reset_index().sort_values(TIME_COL)
    )

    logging.info("Calculating the consecutive item ids for each user")
    kt4_consecutive_df = calculate_consecutive_user_item_ids(lectures_df)

    logging.info("Calculating action gaps")
    kt4_gaps_df = calculate_action_gaps(kt4_consecutive_df)

    logging.info("Loading video_df")
    video_df = pd.read_parquet(ednet_path / "kt4_lectures_estimated.parquet")
    video_df["tags"] = video_df["tags"].fillna(0).astype(int)

    logging.info("Adding video context")
    LECTURE_COLS = ["tags", ITEM_COL, "length"]
    kt4_video = video_df[LECTURE_COLS].merge(kt4_gaps_df, on=ITEM_COL)

    # Spent 28 minutes to run on complete dataset
    logging.info("Generating new session ids per consecutive id")
    kt4_group = kt4_video.groupby([USER_COL, CONSECUTIVE_COL], group_keys=False)
    logging.info("New sessions generation starting")
    kt4_sessions = kt4_video.assign(**{SESSION_COL: kt4_group.apply(generate_sessions)})

    logging.info(
        f"#interactions: {kt4_sessions.shape[0]}, #users: {kt4_sessions[USER_COL].nunique()}"
        f", #items: {kt4_sessions[ITEM_COL].nunique()}"
    )
    logging.info(
        f"#sessions: {kt4_sessions.groupby([USER_COL, CONSECUTIVE_COL])[SESSION_COL].nunique().sum()}"
    )

    logging.info("Removing unnecessary sessions")
    kt4_sessions_clean = kt4_sessions.groupby(
        [USER_COL, CONSECUTIVE_COL], group_keys=False
    ).apply(clean_sessions)

    logging.info(
        f"#interactions: {kt4_sessions_clean.shape[0]}, #users: {kt4_sessions_clean[USER_COL].nunique()}"
        f", #items: {kt4_sessions_clean[ITEM_COL].nunique()}"
    )
    logging.info(
        f"#sessions: {kt4_sessions_clean.groupby([USER_COL, CONSECUTIVE_COL])[SESSION_COL].nunique().sum()}"
    )
    logging.info("Starting blacklist process")
    logging.info("\tGenerating blacklist and cleaning dataset")
    sessions_clean, blacklist = remove_blacklist_users(kt4_sessions_clean)

    logging.info(
        f"\tLocal number of blacklisted users: {blacklist.shape[0]}"
        f" out of {kt4_sessions_clean[USER_COL].nunique()}"
    )
    n_records, cleaned_n_records = kt4_sessions_clean.shape[0], sessions_clean.shape[0]
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
    blacklist_out = Path("pre_processing") / "ednet_blacklists_fix"
    blacklist_out.mkdir(exist_ok=True, parents=True)
    blacklist.to_csv(blacklist_out / f"{'.'.join(in_out_name.split('.')[:-1])}.csv")

    logging.info(
        f"Storing cleaned and sessionified dataset to {sessions_out_path / in_out_name}"
    )
    sessions_clean.to_parquet(sessions_out_path / in_out_name)
    logging.info("COMPLETE")
