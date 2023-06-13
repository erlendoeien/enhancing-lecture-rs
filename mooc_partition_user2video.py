import logging
from pathlib import Path

import dask.dataframe as dd
import pyarrow as pa

from utils.io import setup_logging

SEQ_STRUCT = pa.list_(
    pa.struct(
        [
            (
                "segment",
                pa.list_(
                    pa.struct(
                        [
                            ("end_point", pa.float64()),
                            ("start_point", pa.float64()),
                            ("speed", pa.float64()),
                            ("local_start_time", pa.int64()),
                        ],
                    )
                ),
            ),
            ("video_id", pa.string()),
        ]
    )
)
N_PARTS = 10

if __name__ == "__main__":
    setup_logging("logs/mooc_user2video_partition.log")
    base_path = Path("~/fall_project/MOOCCubeX/")
    results_path = Path("./results")
    relations_path = base_path / "relations"
    out_path = relations_path / "user2video_partitions"

    logging.info("Loading raw user2video dataset")
    user2video = dd.read_json(
        relations_path / "user-video.json", lines=True, blocksize=2**27
    )

    logging.info("Indexing by user_id for partitions")
    user2video_indexed = user2video.set_index("user_id")
    logging.info("Repartitioning dataset")
    user2video_repartitioned = user2video_indexed.repartition(npartitions=N_PARTS)

    logging.info("Divisions")
    logging.info(user2video_repartitioned.divisions)

    logging.info("Dataset memory usage")
    logging.info(f"{user2video_repartitioned.memory_usage_per_partition().compute()}\n")

    logging.info(f"Storing {len(user2video_repartitioned.divisions)+1} partitions")
    # Custom py arrow schema for the sequences
    user2video_repartitioned.to_parquet(out_path, schema={"seq": SEQ_STRUCT})
    logging.info("## COMPLETE ##")
