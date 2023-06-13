SEED = 379

MAX_SEQUENCE_LENGTH = 30
MAX_CONCEPT_LENGTH = 10
ITEM_COL = "item_id"

SEG_REP_COLS = [
    f"seg_rep_{el}" for el in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60]
]
RAW_INT_COLS = ["num_forward", "num_backward", "num_pause"] + SEG_REP_COLS

_FLOAT_COLS = [
    "time_spent",
    "median_pause",
    "time_comp",
    "time_played",
    "replay_length",
    "skipped_length",
]
_MOOC_FLOAT_COLS = ["avg_speed", "std_speed", "eff_speed"]
CONCEPT_COL = "concepts"
FIELD_COL = "fields"
_MOOC_CAT_COLS = [CONCEPT_COL, FIELD_COL]
_EDNET_CAT_COLS = ["tags"]
# Max values retrieved from integer lookup tables in ./embeddings
CONCEPT_MAX = 159318  # 159007
FIELD_MAX = 72

USER_COL = "user_id"
TIME_COL = "timestamp"
ITEM_COL = "item_id"
