import logging
from pathlib import Path

import pandas as pd
from merlin_standard_lib import ColumnSchema, Schema, Tag
from merlin_standard_lib.proto.schema_bp import ValueCount

from utils.constants import (
    _EDNET_CAT_COLS,
    _FLOAT_COLS,
    _MOOC_CAT_COLS,
    _MOOC_FLOAT_COLS,
    CONCEPT_COL,
    CONCEPT_MAX,
    FIELD_COL,
    FIELD_MAX,
    ITEM_COL,
    MAX_CONCEPT_LENGTH,
    MAX_SEQUENCE_LENGTH,
    RAW_INT_COLS,
)


def FLOAT_COLS(dataset: str, is_raw=False) -> list:
    """Loading float columns name. If the dataset is not preprocessed, load
    the initial integer continuous columns as well."""
    cols = _FLOAT_COLS
    if dataset == "mooc":
        cols += _MOOC_FLOAT_COLS
    if not is_raw:
        cols += RAW_INT_COLS
    return cols


def CAT_COLS(dataset: str) -> list:
    if dataset == "ednet":
        return _EDNET_CAT_COLS
    return _MOOC_CAT_COLS


def _create_schema(
    stats_df: pd.DataFrame,
    CAT_FEATS: list,
    INT_CONT_FEATS: list,
    FLOAT_CONT_FEATS: list,
    item_col=ITEM_COL,
) -> Schema:
    """Creates a schema based on the supplied feature names categorized by
    their domain. Should be used initially for a given dataset and later on loaded and
    filtered"""
    item_col = ColumnSchema.create_categorical(
        item_col,
        stats_df.loc["max", item_col].astype(int),
        value_count=ValueCount(0, MAX_SEQUENCE_LENGTH),
        min_index=0,
        tags=[Tag.LIST, Tag.ITEM, Tag.ITEM_ID],
    )
    cat_cols = [
        ColumnSchema.create_categorical(
            col_name,
            stats_df.loc["max", col_name].astype(int),
            shape=(
                (MAX_SEQUENCE_LENGTH, 1, MAX_CONCEPT_LENGTH)
                if col_name in _MOOC_CAT_COLS
                else None
            ),
            value_count=(
                ValueCount(0, MAX_SEQUENCE_LENGTH)
                if col_name not in _MOOC_CAT_COLS
                else None
            ),
            min_index=0,
            tags=[Tag.LIST],
        )
        for col_name in CAT_FEATS
    ]
    int_cont_cols = [
        ColumnSchema.create_continuous(
            col_name,
            min_value=stats_df.loc["min", col_name],
            max_value=stats_df.loc["max", col_name],
            value_count=ValueCount(0, MAX_SEQUENCE_LENGTH),
            tags=[Tag.LIST],
        )
        for col_name in INT_CONT_FEATS
    ]
    float_cont_cols = [
        ColumnSchema.create_continuous(
            col_name,
            min_value=stats_df.loc["min", col_name],
            max_value=stats_df.loc["max", col_name],
            is_float=True,
            value_count=ValueCount(0, MAX_SEQUENCE_LENGTH),
            tags=[Tag.LIST],
        )
        for col_name in FLOAT_CONT_FEATS
    ]
    return Schema([item_col] + cat_cols + int_cont_cols + float_cont_cols)


def create_base_schema(base_path: Path):
    feat_stats = pd.read_parquet(base_path / "feature_stats.parquet")
    # As they are lists of lists -> They are not included in the feature stats df
    feat_stats.loc["max", CONCEPT_COL] = CONCEPT_MAX
    feat_stats.loc["max", FIELD_COL] = FIELD_MAX
    # The current dir name is the feature set type
    is_raw = base_path.name == "raw_dataset"
    int_feats = RAW_INT_COLS if is_raw else list()
    dataset = str(base_path.parent.parent)
    base_schema = _create_schema(
        feat_stats, CAT_COLS(dataset), int_feats, FLOAT_COLS(dataset, is_raw=is_raw)
    )
    with open(base_path / "schema.pb", "w") as file:
        file.write(base_schema.to_proto_text())
    return base_schema


def get_schema(schema_path: str, base_path: Path):
    """Handle Processing and loading of data schema"""
    if schema_path:
        schema_path = Path(schema_path)
    else:
        schema_path = base_path / "schema.pb"

    if schema_path.exists():
        logging.info(f"Loading existing schema {schema_path}")
        base_schema = Schema().from_proto_text(schema_path)
    else:
        logging.info(f"Creating and saving base schema for {base_path}")
        base_schema = create_base_schema(base_path)
    return base_schema
