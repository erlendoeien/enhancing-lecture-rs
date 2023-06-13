import time

import numpy as np
import pandas as pd

TIME_COL = "local_start_time"
END_TIME_COL = "local_end_time"
FW_GAP_COL = "forward_gap"
BACK_GAP_COL = "backward_gap"
USER_COL = "user_id"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"
START_COL = "start"
END_COL = "end"
PAUSE_THRESH = (
    2  # seconds #pd.to_timedelta(2, unit="s") # Same as in In-video behaviour
)

# Heartbeats every 5 seconds recording where the user is
# On average assume that 2.5 seconds watching new segment and previous 2.5 seconds
# in prev segment
# So have to adjust for speed as well
# TODO FIX NEGATIVE INTERVALS - CURRENTLY DROPPED


def get_average_speed(group: pd.DataFrame) -> float:
    """AS, not applicable to EdNet, All non-positive intervals are dropped"""
    if group["duration"].sum() == 0:
        return group["speed"].mean()
    return np.average(group["speed"], weights=group["duration"])


def get_std_speed(group) -> float:
    """Standard deviation of video speed, not weigthed by duration"""
    return group["speed"].std()


def get_eff_video_speed_change(group: pd.DataFrame) -> float:
    """SC - Average speed - minus start speed."""
    return get_average_speed(group) - group["speed"].iat[0]


def get_time_spent(group) -> float:
    """Get the raw, total amount of time spent"""
    return group[END_TIME_COL].iat[-1] - group[TIME_COL].iat[0]


def get_num_pauses(group) -> float:
    """NP, accounting for gaps between intervals larger than some threshold.
    For MOOC: Gaps between intervals, skipping first row's backward's gap
    For EdNet: Num pause actions where the pause lasted longer than thresh"""
    return (group.iloc[1:][BACK_GAP_COL] > PAUSE_THRESH).sum()


# Should add FORWARD GAP feature before
def get_median_pause_dur(group) -> float:
    """Get median pause duration based on gaps between intervals
    larger than pause thresh"""
    skip_first = group.iloc[1:]
    return skip_first[skip_first[BACK_GAP_COL] > PAUSE_THRESH][BACK_GAP_COL].median()


def get_num_ff(group) -> float:
    """NF
    Each interval is interrupted by skip/scrub or pause.
    Check if current start point is larger than previous interval end point
    Do count consecutive play events for instance, though might not be entirely correct
    also in Brinton PLA"""
    return (group[START_COL] > group[END_COL].shift()).sum()


def get_num_backward_seeks(group) -> float:
    """NB -
    Check if current start point is less than previous interval's end point
    Also in Brinton PLA"""
    return (group[START_COL] < group[END_COL].shift()).sum()


def get_raw_intervals(group, start_col=START_COL, end_col=END_COL):
    """Generating intervals of positive play->pause sequences."""
    intervals = group[[start_col, end_col, END_TIME_COL, TIME_COL, "speed", "duration"]]

    # TODO: SHOULD?? Clip cursor value by video length as some are overestimated
    # intervals_clipped.loc[:, [start_col, end_col]] =
    # intervals[[start_col, end_col]].clip(upper=duration)
    return intervals[intervals[start_col] < intervals[end_col]]


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

    part_0 = (
        pd.read_parquet(
            "~/fall_project/MOOCCubeX/relations/sessions_repartitioned/part.0.parquet"
        )
        .reset_index()
        .sort_values(TIME_COL)
    )
    test_data_2_df = part_0[
        (part_0[USER_COL] == "U_10027865")
        & (part_0[SESSION_COL] == 2)
        & (part_0[CONSECUTIVE_COL] == 1)
    ]
    test_data_2_group = test_data_2_df.groupby(
        [USER_COL, CONSECUTIVE_COL, SESSION_COL], group_keys=False
    )

    print("\n" + "#" * 40)
    print("Testing non-interval based features, only on interaction session")
    print("\tNumber of test groups", test_data_2_group.ngroups)
    print("#" * 40)

    print("Testing Time Spent")
    assert (
        (test_time_spent := test_data_2_group.apply(get_time_spent))
        == (exp_spent := 2311)
    ).all(), f"Was {test_time_spent}, should be {exp_spent}"
    print("\t[COMPLETE]")

    print("Testing Number of pauses")
    assert (
        (test_num_pauses := test_data_2_group.apply(get_num_pauses))
        == (exp_num_pauses := 8)
    ).all(), f"Was {test_num_pauses}, should be {exp_num_pauses}"
    print("\t[COMPLETE]")

    print("Testing Median pause duration")
    assert (
        (test_pause_dur_2 := test_data_2_group.apply(get_median_pause_dur)) > 30
    ).all(), (
        "Median pause duration should be more than 1s, was"
        f"{test_pause_dur_2.values[0]}"
    )
    print("\t[COMPLETE]")

    print("Testing Number of forward seeks")
    assert (
        (test_num_ff := test_data_2_group.apply(get_num_ff)) == 8
    ).all(), f"Number of forward seeks should be 8, was {test_num_ff.iat[0]}"
    print("\t[COMPLETE]")

    print("Testing Number of backward seeks")
    assert (
        (test_num_backwards := test_data_2_group.apply(get_num_backward_seeks)) == 0
    ).all(), f"Number of backwards seeks should be 0, was {test_num_backwards.iat[0]}"
    print("\t[COMPLETE]")

    print("Testing speed features")
    assert (
        test_avg_speed := test_data_2_group.apply(get_average_speed).round(2) == 1.40
    ).all(), f"Average speed was {test_avg_speed}, expected ~1.40"
    assert (
        (test_eff_speed := test_data_2_group.apply(get_eff_video_speed_change)).round(2)
        == -0.60
    ).all(), f"Effective speed change was {test_eff_speed}, expected ~ -0.60"
    assert (
        (test_std_speed := test_data_2_group.apply(get_std_speed)).round(2) == 0.44
    ).all(), f"Std speed was {test_std_speed}, expected ~{0.44}"
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
        num_intervals_exp := 9
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
        "time_spent": get_time_spent,
        "num_forward": get_num_ff,
        "num_backward": get_num_backward_seeks,
        "num_pause": get_num_pauses,
        "median_pause": get_median_pause_dur,
        "avg_speed": get_average_speed,
        "std_speed": get_std_speed,
        "eff_speed_change": get_eff_video_speed_change,
    }

    part_0_group = part_0[:1000].groupby(
        [USER_COL, CONSECUTIVE_COL, SESSION_COL], group_keys=False
    )
    start_time = time.time()
    part_seq = test_data_2_group.progress_apply(
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
    print(part_seq.describe())
