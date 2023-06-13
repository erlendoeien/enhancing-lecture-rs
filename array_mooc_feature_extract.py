import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocesspandas import applyparallel  # noqa

from feature_extractors import (
    get_replay_length,
    get_seg_reps,
    get_session_features,
    get_skipped_length,
    get_time_comp,
    get_time_played,
)
from mooc_feature_extractors import (
    get_average_speed,
    get_eff_video_speed_change,
    get_median_pause_dur,
    get_num_backward_seeks,
    get_num_ff,
    get_num_pauses,
    get_raw_intervals,
    get_std_speed,
    get_time_spent,
)
from utils.io import setup_logging

USER_COL = "user_id"
TIME_COL = "local_start_time"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"

group_extract_dict = {
    "time_spent": get_time_spent,
    "num_forward": get_num_ff,
    "num_backward": get_num_backward_seeks,
    "num_pause": get_num_pauses,
    "median_pause": get_median_pause_dur,
    "std_speed": get_std_speed,
    "avg_speed": get_average_speed,
    "eff_speed": get_eff_video_speed_change,
}

# Add SegRep ablation study
OVERLAP_THRESH_SECONDS = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60])
interval_extract_dict = {
    f"seg_rep_{thresh}": lambda group, overlap_thresh=thresh: get_seg_reps(
        group, overlap_thresh_ms=overlap_thresh
    )
    for thresh in OVERLAP_THRESH_SECONDS
}
interval_extract_dict = {
    **interval_extract_dict,
    **{
        "time_comp": get_time_comp,
        "time_played": get_time_played,
        "replay_length": get_replay_length,
        "skipped_length": get_skipped_length,
    },
}


if __name__ == "__main__":
    partition_num = int(sys.argv[1])
    setup_logging(
        Path("logs") / "mooc_extract_features_fix" / f"part.{partition_num}.log"
    )

    base_path = Path("~/fall_project/MOOCCubeX/")
    relations_path = base_path / "relations"
    sessions_in_path = relations_path / "sessions_repartitioned_fix"
    sessions_out_path = relations_path / "sessions_featured_fix"
    sessions_out_path.mkdir(exist_ok=True, parents=True)

    in_out_name = f"part.{partition_num}.parquet"

    logging.info(f"Loading partition {in_out_name}")
    sessions_df = pd.read_parquet(sessions_in_path / in_out_name).sort_values(TIME_COL)
    logging.info("Removing all non-positive intervals")
    sessions_df = sessions_df[sessions_df["start"] < sessions_df["end"]]
    sessions_group = sessions_df.groupby([USER_COL, CONSECUTIVE_COL, SESSION_COL])

    logging.info("Extracting session features")
    extracted_features = sessions_group.apply_parallel(
        get_session_features,
        get_intervals_func=get_raw_intervals,
        group_extract_dict=group_extract_dict,
        interval_extract_dict=interval_extract_dict,
    )

    logging.info("Merging features with first interaction in each interaction session")
    sessions_w_features = extracted_features.merge(
        sessions_group.first(), left_index=True, right_index=True
    )

    logging.info(f"Storing sesssions with extra to {sessions_out_path / in_out_name}")
    sessions_w_features.to_parquet(sessions_out_path / in_out_name)
    logging.info("COMPLETE")
