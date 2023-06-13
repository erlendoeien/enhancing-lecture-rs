from functools import partial
from pathlib import Path

import torch
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt

from utils.constants import _MOOC_CAT_COLS, ITEM_COL
from utils.io import load_embedding_table

# Enable one-hot since labels have to be 1-dimensional and loader failes to
# make them so
# DEFAULT_METRICS = list(get_metrics(ks=[4, 10], one_hot=True).values())
DEFAULT_METRICS = [
    NDCGAt(top_ks=[5, 10], labels_onehot=True),
    RecallAt(top_ks=[5, 10], labels_onehot=True),
    AvgPrecisionAt(top_ks=[5, 10], labels_onehot=True),
]


def get_next_item_pred_task(
    label_smoothing_factor: float,
    tie_weights=True,
    metrics=DEFAULT_METRICS,
):
    label_smoothing_xe_loss = tr.LabelSmoothCrossEntropyLoss(
        reduction="mean", smoothing=label_smoothing_factor
    )

    return tr.NextItemPredictionTask(
        weight_tying=tie_weights, metrics=metrics, loss=label_smoothing_xe_loss
    )


def create_input_module_args(base_config: dict, columns: list, is_trainable=False):
    """Currently only for item_id, must be udpated to include other features"""
    base = dict()
    # Handle item id embedding dim, std and input dropout
    base["embedding_dims"] = {
        ITEM_COL: (item_embed_dim := base_config.pop("item_id_embedding_dim"))
    }
    # Setting all other categorical embeddings equal to item embedding
    # Since infer is set -> Won't actually have an impact
    base["embedding_dim_default"] = item_embed_dim

    item_std = base_config.pop("item_id_embedding_init_std")
    other_std = base_config.pop("other_embedding_init_std", 0)
    if not other_std and len(columns) > 1:
        raise RuntimeError(
            Exception(f"Missing other embedding init std, but supplied {columns}")
        )

    # Configuring embedding initializers
    embeddings_initializers = {}
    # If other_std is not defined -> Should contain only ITEM_COL
    for col in columns:
        if col == ITEM_COL:
            std = item_std
        else:
            std = other_std
        embeddings_initializers[col] = partial(torch.nn.init.normal_, mean=0.0, std=std)

    # Add pretrained embeddings for MOOC columns
    if all(col in columns for col in _MOOC_CAT_COLS) and (
        base_config.pop("pretrained_embeddings")
    ):
        pretrained_embeddings_init = {
            col: tr.PretrainedEmbeddingsInitializer(
                load_embedding_table(Path("embeddings"), col),
                trainable=is_trainable,
            )
            for col in _MOOC_CAT_COLS
            if col in columns
        }
        # Have to add the pretrained embedding dims - Fine to override
        pretrained_embedding_dims = {
            col: pretrained_init.weight_matrix.shape[1]
            for col, pretrained_init in pretrained_embeddings_init.items()
        }

        base["embedding_dims"] = {**base["embedding_dims"], **pretrained_embedding_dims}
        embeddings_initializers.update(pretrained_embeddings_init)

    base["embeddings_initializers"] = embeddings_initializers
    if (input_dropout := base_config.pop("input_dropout")) > 0:
        base_config["post"].append(tr.TabularDropout(dropout_rate=input_dropout))
    base.update(base_config)
    return base
    return base
