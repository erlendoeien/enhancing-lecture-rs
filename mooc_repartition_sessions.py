import logging
from pathlib import Path

import dask.dataframe as dd
import pyarrow as pa

from utils.io import setup_logging

PARTS_OUT = 8
if __name__ == "__main__":
    setup_logging("logs/mooc_repartition.log")
    base_path = Path("~/fall_project/MOOCCubeX/")
    relations_path = base_path / "relations"
    out_path = relations_path / "sessions_repartitioned_fix"
    out_path.mkdir(exist_ok=True, parents=True)

    logging.info("Loading session files")
    sessions = dd.read_parquet(relations_path / "sessions_with_context_fix")

    logging.info("Renaming interval columns")
    sessions_renamed = sessions.rename(
        columns={"start_point": "start", "end_point": "end", "video_id": "item_id"}
    )

    logging.info("Shuffling by user_id for partitions")
    sessions_indexed = sessions_renamed.set_index("user_id")
    sessions_shuffled = sessions_indexed.shuffle(
        on=sessions_indexed.index, npartitions=PARTS_OUT
    )

    logging.info("Divisions")
    logging.info(sessions_shuffled.divisions)

    logging.info("Dataset memory usage")
    logging.info(f"\n{sessions_shuffled.memory_usage_per_partition().compute()}\n")

    logging.info(f"Storing {len(sessions_shuffled.divisions)-1} partitions")
    # Custom py arrow schema for the sequences
    sessions_shuffled.to_parquet(out_path, schema={"concept_id": pa.list_(pa.string())})
    logging.info("## COMPLETE ##")
