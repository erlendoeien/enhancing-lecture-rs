from transformers4rec.torch.ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt


def get_metrics(ks=[5, 10], one_hot=False):
    return {
        "map": AvgPrecisionAt(top_ks=ks, labels_onehot=one_hot),
        "recall": RecallAt(top_ks=ks, labels_onehot=one_hot),
        "ndcg": NDCGAt(top_ks=ks, labels_onehot=one_hot),
    }
