import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_latest_checkpoint(directory):
    checkpoints = glob.glob(os.path.join(directory, "checkpoint-*"))
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def get_log_history(directory):
    checkpoint_dir = get_latest_checkpoint(directory)
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")

    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    log_history = trainer_state["log_history"]
    return log_history


def get_train_history_dataset(paths):
    model_history = {}
    for model_dir in paths:
        model = "_".join(model_dir.stem.split("_")[1:-3])
        seed_entries = {}
        for seed_folder in model_dir.glob("seed_*"):
            log_history = get_log_history(seed_folder)
            seed = int(seed_folder.stem.split("_")[1])
            train_epoch, val_epoch = pd.DataFrame.from_dict(
                log_history[::2]
            ), pd.DataFrame.from_dict(log_history[1::2])
            seed_entries[seed] = train_epoch.merge(
                val_epoch, on=["epoch", "step"], how="outer"
            )

        model_history[model] = pd.concat(
            seed_entries.values(), keys=seed_entries.keys(), names=["seed"]
        ).rename(columns={"loss": "Val", "eval_/loss": "Test", "epoch": "Epoch"})[
            ["Epoch", "learning_rate", "step", "Val", "Test"]
        ]
        # print(model_history)
    return (
        pd.concat(model_history.values(), keys=model_history.keys(), names=["Model"])
        .droplevel(-1)
        .set_index("Epoch", append=True)
    )


def plot_loss_w_uncertainty(df, dataset, model="bert"):
    df.index = df.index.map(
        lambda name: (
            " ".join([tok.capitalize() for tok in name[0].split("_")[1:]]).strip(),
            name[1],
            name[2],
        )
    )
    df = df.rename(index={"Bias Adj": "Bias-Adj"})
    melted = df.reset_index().melt(
        id_vars=["Model", "Epoch"],
        value_vars=["Val", "Test"],
        var_name="Loss Type",
        value_name="Loss",
    )
    fig, ax = plt.subplots()
    sns.lineplot(
        melted,
        x="Epoch",
        y="Loss",
        hue="Model",
        style="Loss Type",
        palette="Blues" if dataset == "ednet" else "Oranges",
        hue_order=["Base", "Full", "Bias-Adj"],
        style_order=["Val", "Test"],
        markers=True,
        ax=ax,
    )
    ax.legend(ncols=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(f"diagrams/{model}_loss_{dataset}.svg")
    fig.savefig(f"diagrams/{model}_loss_{dataset}.pdf")


def test_equal_num_eochs(df):
    num_epochs = (
        df.reset_index("Epoch").groupby(["Model", "seed"], group_keys=False).size()
    )
    assert (
        num_epochs.groupby(["Model"]).apply(
            lambda group: ((group.iloc[0] == group).all() & (group.shape[0] == 10))
        )
    ).all()


models = ["bert", "gru", "xlnet"]
plt.rcParams.update({"font.size": 14})
for model in models:
    print("Plotting model", model)
    mooc_paths = Path(".").glob(f"mooc_{model}*")
    ednet_paths = Path(".").glob(f"ednet_{model}*")
    mooc_df = get_train_history_dataset(mooc_paths)
    ednet_df = get_train_history_dataset(ednet_paths)
    test_equal_num_eochs(mooc_df)
    test_equal_num_eochs(ednet_df)
    plot_loss_w_uncertainty(mooc_df, "mooc", model=model)
    plot_loss_w_uncertainty(ednet_df, "ednet", model=model)


# mooc_bert_paths = Path(".").glob("mooc_bert*")
# ednet_bert_paths = Path(".").glob("ednet_bert*")

# mooc_df = get_train_history_dataset(mooc_bert_paths)
# ednet_df = get_train_history_dataset(ednet_bert_paths)

# # print(*list(mooc_bert_paths), sep="\n")
# # print(*list(ednet_bert_paths), sep="\n")
# print(mooc_df)
# plot_loss_w_uncertainty(mooc_df, "mooc")
# print(ednet_df)
# plot_loss_w_uncertainty(ednet_df, "ednet")
