import logging
from pathlib import Path

import pandas as pd

from utils.io import setup_logging

if __name__ == "__main__":
    setup_logging("logs/mooc_estimate_video_length.log")
    base_path = Path("~/fall_project/MOOCCubeX/")
    entities_path = base_path / "entities"
    relations_path = base_path / "relations"
    out_path = entities_path / "video_lengths.parquet"

    logging.info("Loading video df")
    video_df = pd.read_json(entities_path / "video.json", lines=True)

    logging.info(
        "Getting the second largest caption end-point to estimate video length"
    )
    video_df["length"] = video_df["end"].transform(
        lambda lst: (sorted(lst, reverse=True) * 2)[1]
    )

    logging.info("Loading concept2video")
    concept2video = pd.read_csv(
        relations_path / "concept-video.txt", names=["concept_id", "ccid"], sep="\t"
    )
    logging.info("Merging in concepts")
    concept_video = concept2video.merge(video_df, on="ccid", how="outer")
    logging.info("Listify concepts")
    context_by_ccid = (
        concept_video.fillna("")
        .groupby("ccid")
        .agg({"name": "first", "length": "first", "concept_id": list})
    )
    logging.info(f"Storing video lengths as {out_path}")
    context_by_ccid[["name", "length", "concept_id"]].to_parquet(out_path)

    logging.info("## COMPLETE ##")
