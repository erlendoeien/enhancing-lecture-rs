import time

import pandas as pd

TIME_COL = "timestamp"
FW_GAP_COL = "forward_gap"
USER_COL = "user_id"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"
PLAY_TYPE = "play_video"
PAUSE_TYPE = "pause_video"
START_COL = "start"
END_COL = "end"
PAUSE_THRESH = pd.to_timedelta(2, unit="s")  # Same as in In-video behaviour


# NONE-INTERVAL-BASED Features
def get_time_spent(group):
    """Calculate the total time spent on a video in real time, including pauses.
    The last action is assumed to be the exiting one, generally speaking"""
    # Convert forward gap to ms
    return group[TIME_COL].iat[-1] - group[TIME_COL].iat[0]


def get_num_pauses(group, action_col="action_type", pause_label="pause_video"):
    """NP, Num pauses larger than threshold"""
    # Assuming few consecutive pause actions, though some, i.e. when scrubbing
    # -> Not actually a pause
    return (
        (group[action_col] == pause_label) & (group[FW_GAP_COL] > PAUSE_THRESH)
    ).sum()


# Should add FORWARD GAP feature before
def get_median_pause_dur(
    group, action_col="action_type", pause_label="pause_video", pause_col=FW_GAP_COL
):
    """MP"""
    return pd.to_timedelta(
        group.loc[group[action_col] == pause_label, pause_col].median(), unit="ms"
    ).round("100ms")


def get_num_ff(group, cursor_col="cursor_time", action_col="action_type"):
    """NF
    Ignoring global timestamps and speeds. Removing border events due
    to inconsistencies in tracking, i.e. sometimes placed before last pause.
    Do count consecutive play events for instance, though might not be entirely correct
    also in Brinton PLA"""
    group_clean = group[~group["action_type"].isin(["enter", "quit"])]
    group_shifted = group_clean.shift(-1)
    is_forward = group_clean[cursor_col] < group_shifted[cursor_col]
    is_linear = (group[action_col] == "play_video") & (
        group_shifted[action_col] == "pause_video"
    )
    return (is_forward & (~is_linear)).sum()


def get_num_backward_seeks(group, cursor_col="cursor_time"):
    """NB - Important factor
    Only basing on cursor time, since we don't have the playtime speed
    Also in Brinton PLA"""
    # Might have to ignore enter or quits because they get in the way
    group_clean = group[~group["action_type"].isin(["enter", "quit"])]
    return (group_clean[cursor_col] < group_clean[cursor_col].shift()).sum()


def get_raw_intervals(
    group,
    cursor_col="cursor_time",
    dur_col=FW_GAP_COL,
    time_col=TIME_COL,
    action_col="action_type",
    start_col="start",
    end_col="end",
):
    """Generating intervals of positive play->pause sequences"""
    # If the current row is play and next is pause OR current row is pause
    #  and previous was play
    is_linear = (
        (group[action_col] == "play_video")
        & (group[action_col].shift(-1) == "pause_video")
    ) | (
        (group[action_col] == "pause_video")
        & (group[action_col].shift() == "play_video")
    )
    group_clean = group[is_linear]

    # Get only the play->pause relations
    intervals = pd.concat(
        [
            group_clean[[cursor_col, dur_col, time_col]].rename(
                columns={cursor_col: "start"}
            ),
            group_clean[cursor_col].shift(-1).rename("end"),
        ],
        axis=1,
    )[::2]

    # TODO: NO LONGER - Clip cursor value by video length as some are overestimated
    intervals_clipped = intervals.copy()
    # intervals_clipped.loc[:, [start_col, end_col]] =
    # intervals[[start_col, end_col]].clip(upper=duration)

    # Removing skip-back intervals and intervals where both start and end are clipped
    # Will remove some information, but less than with
    # removing the lectures all together
    return intervals_clipped[intervals_clipped[start_col] < intervals_clipped[end_col]]


if __name__ == "__main__":
    test_range_array_1 = pd.arrays.IntervalArray.from_arrays([0, 3, 5], [1, 4, 6])  # 0
    test_range_array_2 = pd.arrays.IntervalArray.from_arrays([0, 3, 5], [4, 4, 6])  # 1
    test_range_array_3 = pd.arrays.IntervalArray.from_arrays([0, 1, 5], [4, 6, 6])  # 2
    test_range_array_4 = pd.arrays.IntervalArray.from_arrays(
        [0, 15, 5, 14, 0], [4, 17, 8, 16, 15]
    )  # 4
    test_range_array_5 = pd.arrays.IntervalArray.from_arrays(
        [0, 2, 5, 8, 1], [2, 3, 8, 16, 4]
    )  # 2
    test_range_array_6 = pd.arrays.IntervalArray.from_arrays([0, 1], [1, 2])

    test_data_2_df = pd.read_parquet("test/test_data_2.parquet")
    test_data_2_group = test_data_2_df.groupby(
        [USER_COL, CONSECUTIVE_COL, SESSION_COL], group_keys=False
    )
    test_data_dur = test_data_2_df["length"].values[0]

    print("\n" + "#" * 40)
    print("Testing non-interval based features, only on interaction session")
    print("\tNumber of test groups", test_data_2_group.ngroups)
    print("#" * 40)

    print("Testing Fraction Spent")
    assert (
        (test_frac_spent := test_data_2_group.apply(get_time_spent))
        == (exp_spent := 1541313117180 - 1541312707824)
    ).all(), f"Was {test_frac_spent}, should be {exp_spent}"
    print("\t[COMPLETE]")

    print("Testing Number of pauses")
    assert (
        (test_num_pauses := test_data_2_group.apply(get_num_pauses))
        == (exp_num_pauses := 4)
    ).all(), f"Was {test_num_pauses}, should be {exp_num_pauses}"
    print("\t[COMPLETE]")

    print("Testing Median pause duration")
    assert (
        (test_pause_dur_2 := test_data_2_group.apply(get_median_pause_dur))
        > pd.Timedelta("00:00:01")
    ).all(), (
        "Median pause duration should be more than 1s, was"
        f"{test_pause_dur_2.values[0].astype('timedelta64[ms]')}"
    )
    print("\t[COMPLETE]")

    print("Testing Number of forward seeks")
    assert (
        (test_num_ff := test_data_2_group.apply(get_num_ff)) == 15
    ).all(), f"Number of forward seeks should be 15, was {test_num_ff.iat[0]}"
    print("\t[COMPLETE]")

    print("Testing Number of backward seeks")
    assert (
        (test_num_backwards := test_data_2_group.apply(get_num_backward_seeks)) == 3
    ).all(), f"Number of backwards seeks should be 3, was {test_num_backwards.iat[0]}"
    print("\t[COMPLETE]")

    print("\n" + "#" * 40)
    print("Testing Interval generator")
    print("#" * 40)
    test_raw_intervals = test_data_2_group.apply(get_raw_intervals)

    # All positive intervals
    assert (
        test_raw_intervals["start"] < test_raw_intervals["end"]
    ).all(), "Expected all intervals to be positive"
    # Number of intervals
    assert (test_num_intervals := test_raw_intervals.shape[0]) == (
        num_intervals_exp := 11
    ), f"Was {test_num_intervals}, expected {num_intervals_exp}"

    # Intervals are monotonically increasing wrt. time
    assert test_raw_intervals[
        TIME_COL
    ].is_monotonic_increasing, "Intervals are not correctly order wrt. time"

    # Returns a df
    assert isinstance(
        test_raw_intervals, pd.DataFrame
    ), f"Was {type(test_num_intervals)}, expected {pd.DataFrame}"
    print("\t[COMPLETE]")

    print("\n" + "#" * 40)
    print("Testing combined aggregate")
    print("#" * 40)
    from multiprocesspandas import applyparallel  # noqa
    from tqdm import tqdm

    from feature_extractors import get_session_features

    tqdm.pandas()

    group_extract_dict = {
        "frac_spent": get_time_spent,
        "num_forward": get_num_ff,
        "num_backward": get_num_backward_seeks,
        "num_pause": get_num_pauses,
        "median_pause": get_median_pause_dur,
    }

    part_0 = pd.read_parquet(
        "~/fall_project/EdNet/KT4_sessions_new/part.0.parquet"
    ).sort_values(TIME_COL)
    part_0_group = part_0[:1000].groupby(
        [USER_COL, CONSECUTIVE_COL, SESSION_COL], group_keys=False
    )
    start_time = time.time()
    part_seq = part_0_group.progress_apply(
        get_session_features,
        get_intervals_func=get_raw_intervals,
        group_extract_dict=group_extract_dict,
    )
    print(f"Apply sequential took: {time.time() - start_time} seconds")
    start_time = time.time()
    part_parallel = part_0_group.apply_parallel(
        get_session_features,
        get_intervals_func=get_raw_intervals,
        group_extract_dict=group_extract_dict,
    )
    print(f"Apply Parallel took: {time.time() - start_time} seconds")
    print(part_seq.describe())
    print(f"Apply Parallel took: {time.time() - start_time} seconds")
    print(part_seq.describe())
