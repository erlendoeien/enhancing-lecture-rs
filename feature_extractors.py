import itertools
import time

import pandas as pd
import piso

USER_COL = "user_id"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"
EDNET_TIME_COL = "timestamp"
MOOC_TIME_COL = "local_start_time"


def _intervals_to_arr(
    raw_intervals: pd.DataFrame, sort=True, start_col="start", end_col="end", **kwargs
) -> pd.arrays.IntervalArray:
    """Helper method to convert raw DF intervals to IntervalArray"""
    interval_arr: pd.arrays.IntervalArray = pd.arrays.IntervalArray.from_arrays(
        raw_intervals[start_col], raw_intervals[end_col], closed="right", **kwargs
    )
    if sort:
        interval_arr = interval_arr[interval_arr.argsort()]
    return interval_arr


def combined_interval_apply(intervals: pd.arrays.IntervalArray, extract_dict=None):
    if not extract_dict:
        return pd.Series(dtype=float)
    if intervals.shape[0] == 0:
        return pd.Series({key: 0 for key in extract_dict}, index=extract_dict.keys())
    features = {
        name: func(
            intervals,
        )
        for name, func in extract_dict.items()
    }
    return pd.Series(features, index=features.keys())


def combined_group_apply(group, extract_dict=None):
    if not extract_dict:
        return pd.Series(dtype=float)
    if group.shape[0] == 0:
        return pd.Series({key: 0 for key in extract_dict}, index=extract_dict.keys())
    features = {name: func(group) for name, func in extract_dict.items()}
    return pd.Series(features, index=features.keys())


def get_session_features(
    group, get_intervals_func, group_extract_dict=None, interval_extract_dict=None
):
    """Method for making watched intervals from the records.
    Should return the extracted feature, often a singular row pd.Series
    NB: Ignoring consecutive play and pause events, which do happen sometimes
    UDF - So use it with Apply"""
    group_features = combined_group_apply(group, extract_dict=group_extract_dict)
    raw_intervals = get_intervals_func(group)
    intervals = _intervals_to_arr(raw_intervals)
    interval_features = combined_interval_apply(
        intervals, extract_dict=interval_extract_dict
    )
    return pd.concat([group_features, interval_features], axis=0)


def get_skipped_length(
    intervals: pd.arrays.IntervalArray,
    start_col="start",
    end_col="end",
):
    # To avoid counting skipped sections which are rewatched later on
    cont_intervals = piso.union(intervals)
    end_times = pd.Series([interval.right for interval in cont_intervals])
    start_times = pd.Series([interval.left for interval in cont_intervals])
    skipped_intervals = pd.concat(
        [end_times.rename(start_col), start_times.shift(-1).rename(end_col)], axis=1
    ).dropna()
    return (skipped_intervals[end_col] - skipped_intervals[start_col]).sum()


def get_time_comp(intervals: pd.arrays.IntervalArray, squeeze=False):
    """Fraction Complete Helper function"""
    merged_intervals = piso.union(intervals, squeeze=squeeze)
    return sum(merged_intervals.length)


def get_time_played(
    intervals: pd.arrays.IntervalArray,
):
    """Helper function for get_time_played"""
    return sum(intervals.length.to_list())


def get_replay_length(intervals: pd.arrays.IntervalArray):
    """Replay length helper function"""
    # If non-overlap -> Will be negative -> Return 0
    # If overlap: Find the intersection and return it
    return sum(
        [
            max(0, min(interval.right, other.right) - max(interval.left, other.left))
            for interval, other in itertools.combinations(intervals, 2)
        ]
    )


def get_seg_reps(intervals: pd.arrays.IntervalArray, overlap_thresh_ms=0):
    """Segment rep helper function"""
    return sum(
        [
            max(interval.left, other.left)
            < (min(interval.right, other.right) - overlap_thresh_ms)
            for interval, other in itertools.combinations(intervals, 2)
        ]
    )


# Previous method, not accounting for thresholds
def count_seg_reps_test(intervals):
    return sum(
        [
            intervals[i + 1 :].overlaps(curr_interval).sum()
            for i, curr_interval in enumerate(intervals)
        ]
    )


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

    print("\n" + "#" * 40)
    print("\tNumber of test groups", test_data_2_group.ngroups)
    print("#" * 40)

    print("\n" + "#" * 40)
    print("Testing interval-based features")
    print("#" * 40)

    print("Testing Segment repetition counting, with threshold")
    assert (
        test_thresh_1 := get_seg_reps(test_range_array_1, overlap_thresh_ms=1)
    ) == 0, f"Was {test_thresh_1}, should be 0"
    assert (
        test_thresh_2 := get_seg_reps(test_range_array_2, overlap_thresh_ms=1)
    ) == 0, f"Was {test_thresh_2}, should be 0"
    assert (
        test_thresh_3 := get_seg_reps(test_range_array_3, overlap_thresh_ms=1)
    ) == 1, f"Was {test_thresh_3}, should be 1"
    assert (
        test_thresh_4 := get_seg_reps(test_range_array_4, overlap_thresh_ms=2)
    ) == 2, f"Was {test_thresh_4}, should be 2"
    assert (
        test_thresh_5 := get_seg_reps(test_range_array_5, overlap_thresh_ms=3)
    ) == 0, f"Was {test_thresh_5}, should be 0"
    assert (
        test_thresh_5 := get_seg_reps(test_range_array_6, overlap_thresh_ms=0)
    ) == 0, f"Was {test_thresh_5}, should be 0"
    print("\t[COMPLETE]")

    print("Testing Segment repetition counting, without a threshold")
    assert (
        test_1 := count_seg_reps_test(test_range_array_1)
    ) == 0, f"Was {test_1}, should be 0"
    assert (
        test_2 := count_seg_reps_test(test_range_array_2)
    ) == 1, f"Was {test_2}, should be 1"
    assert (
        test_3 := count_seg_reps_test(test_range_array_3)
    ) == 2, f"Was {test_3}, should be 2"
    assert (
        test_4 := count_seg_reps_test(test_range_array_4)
    ) == 4, f"Was {test_4}, should be 4"
    assert (
        test_5 := count_seg_reps_test(test_range_array_5)
    ) == 2, f"Was {test_5}, should be 2"
    assert (
        test_5 := count_seg_reps_test(
            pd.arrays.IntervalArray.from_arrays([0, 1], [1, 2])
        )
    ) == 0, f"Was {test_5}, should be 0"
    print("\t[COMPLETE]")

    print("Testing Length completed")
    assert (
        comp_1 := get_time_comp(test_range_array_1)
    ) == 3, f"Was {comp_1}, should be 3"
    assert (
        comp_2 := get_time_comp(test_range_array_2)
    ) == 5, f"Was {comp_2}, should be 5"
    assert (
        comp_3 := get_time_comp(test_range_array_3)
    ) == 6, f"Was {comp_3}, should be 6"
    assert (
        comp_4 := get_time_comp(test_range_array_4)
    ) == 17, f"Was {comp_4}, should be 17"
    assert (comp_5 := get_time_comp(test_range_array_5)) == (
        4 + 11
    ), f"Was {comp_5}, should be 15"
    print("\t[COMPLETE]")

    print("Testing length Played")
    assert (
        played_1 := get_time_played(test_range_array_1)
    ) == 3, f"Was {played_1}, should be {3}"
    assert (
        played_2 := get_time_played(test_range_array_2)
    ) == 6, f"Was {played_2}, should be {7}"
    assert (
        played_3 := get_time_played(test_range_array_3)
    ) == 10, f"Was {played_3}, should be {10}"
    assert (
        played_4 := get_time_played(test_range_array_4)
    ) == 26, f"Was {played_4}, should be {26}"
    assert (
        played_5 := get_time_played(test_range_array_5)
    ) == 17, f"Was {played_5}, should be {17}"
    print("\t[COMPLETE]")

    print("Testing Replay length")
    assert (
        replay_1 := get_replay_length(test_range_array_1)
    ) == 0, f"Was {replay_1}, should be {0}"
    assert (
        replay_2 := get_replay_length(test_range_array_2)
    ) == 1, f"Was {replay_2}, should be {1}"
    assert (
        replay_3 := get_replay_length(test_range_array_3)
    ) == 4, f"Was {replay_3}, should be {4}"
    assert (
        replay_4 := get_replay_length(test_range_array_4)
    ) == 9, f"Was {replay_4}, should be {9}"
    assert (
        replay_5 := get_replay_length(test_range_array_5)
    ) == 2, f"Was {replay_5}, should be {2}"
    print("\t[COMPLETE]")

    print("Testing skip length features")
    assert (
        skip_1 := get_skipped_length(test_range_array_1)
    ) == 3, f"Was {skip_1}, should be {3}"
    assert (
        skip_2 := get_skipped_length(test_range_array_2)
    ) == 1, f"Was {skip_2}, should be {1}"
    assert (
        skip_3 := get_skipped_length(test_range_array_3)
    ) == 0, f"Was {skip_3}, should be {0}"
    assert (
        skip_4 := get_skipped_length(test_range_array_4)
    ) == 0, f"Was {skip_4}, should be {0}"
    assert (
        skip_5 := get_skipped_length(test_range_array_5)
    ) == 1, f"Was {replay_5}, should be {1}"
    print("\t[COMPLETE]")

    print("\n" + "#" * 40)
    print("Testing combined aggregate")
    print("#" * 40)
    import importlib

    from multiprocesspandas import applyparallel  # noqa
    from tqdm import tqdm

    tqdm.pandas()

    interval_extract_dict = {
        "seg_rep": get_seg_reps,
        "length_comp": get_time_comp,
        "length_played": get_time_played,
        "replay_length": get_replay_length,
        "get_skipped_length": get_skipped_length,
    }
    ednet_test_part = "~/fall_project/EdNet/KT4_sessions_new/part.0.parquet"
    mooc_test_part = (
        "~/fall_project/MOOCCubeX/relations/sessions_repartitioned/part.0.parquet"
    )
    mooc_extractor = importlib.import_module("mooc_feature_extractors")
    ednet_extractor = importlib.import_module("ednet_feature_extractors")

    def test_group_apply(partition_path: str, time_col, extract_func, nrows=1000):
        part_0 = pd.read_parquet(partition_path).sort_values(time_col)
        part_0_group = part_0[:nrows].groupby(
            [USER_COL, CONSECUTIVE_COL, SESSION_COL], group_keys=False
        )
        start_time = time.time()
        part_seq = part_0_group.progress_apply(
            get_session_features,
            get_intervals_func=extract_func,
            interval_extract_dict=interval_extract_dict,
        )
        print(f"Apply sequential took: {time.time() - start_time} seconds")
        start_time = time.time()
        part_0_group.apply_parallel(
            get_session_features,
            get_intervals_func=extract_func,
            interval_extract_dict=interval_extract_dict,
        )
        print(f"Apply Parallel took: {time.time() - start_time} seconds")
        print(part_seq.describe())

    print("Testing Ednet part 0")
    test_group_apply(
        ednet_test_part, EDNET_TIME_COL, extract_func=ednet_extractor.get_raw_intervals
    )
    print("\nTesting mooc part 0")
    test_group_apply(
        mooc_test_part, MOOC_TIME_COL, extract_func=mooc_extractor.get_raw_intervals
    )
