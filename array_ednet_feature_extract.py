import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocesspandas import applyparallel  # noqa

from ednet_feature_extractors import (
    get_median_pause_dur,
    get_num_backward_seeks,
    get_num_ff,
    get_num_pauses,
    get_raw_intervals,
    get_time_spent,
)
from feature_extractors import (
    get_replay_length,
    get_seg_reps,
    get_session_features,
    get_skipped_length,
    get_time_comp,
    get_time_played,
)
from utils.io import setup_logging

ITEM_COL = "item_id"
USER_COL = "user_id"
TIME_COL = "timestamp"
SESSION_COL = "session_id"
CONSECUTIVE_COL = "item_consecutive_id"
BACK_GAP_COL = "back_gap"
FW_GAP_COL = "forward_gap"

group_extract_dict = {
    "time_spent": get_time_spent,
    "num_forward": get_num_ff,
    "num_backward": get_num_backward_seeks,
    "num_pause": get_num_pauses,
    "median_pause": get_median_pause_dur,
}

# Add SegRep ablation study
OVERLAP_THRESH_SECONDS = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60])
interval_extract_dict = {
    f"seg_rep_{thresh}": lambda group, overlap_thresh=thresh: get_seg_reps(
        group, overlap_thresh_ms=overlap_thresh
    )
    for thresh in (OVERLAP_THRESH_SECONDS * 1000)
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
        Path("logs") / "ednet_extract_features_fix" / f"part.{partition_num}.log"
    )

    ednet_path = Path("~/fall_project/EdNet")
    sessions_path = ednet_path / "KT4_sessions_fix"
    feature_out_path = ednet_path / "KT4_session_features_fix"

    in_out_name = f"part.{partition_num}.parquet"

    logging.info(f"Loading partition {in_out_name}")
    sessions_df = pd.read_parquet(sessions_path / in_out_name).sort_values(TIME_COL)
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

    logging.info(f"Storing sesssions with extra to {feature_out_path / in_out_name}")
    sessions_w_features.to_parquet(feature_out_path / in_out_name)
    logging.info("COMPLETE")
