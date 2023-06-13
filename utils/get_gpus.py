from pathlib import Path

import pandas as pd


def get_gpu_from_log(path):
    with open(path) as f:
        gpu_line = f.readline().strip()

    return " ".join(gpu_line.split(" ")[5:])


def parse_name(path):
    tokens = path.name.split("_")
    start = 1
    end = -3
    if "baseline" in path.name:
        start = 2
        end = -2
    return "_".join(tokens[start:end])


model_name_map = {"gru": "GRU", "bert": "BERT", "xlnet": "XLNet"}
variant_name_map = {
    "base": "Base",
    "full": "Full",
    "bias_adj": "Bias Adj",
    "features": "Features",
}

baseline_order = ["iALS", "LMF", "BPR", "KNN"]
# mooc_baseline_order = ednet_baseline_order[:2] + ["Syllabus"] + ednet_baseline_order[2:]
sars_order = ["gru", "bert", "xlnet"]
sars_base_order = [f"{model}_base" for model in sars_order]
sars_full_order = [f"{model}_full" for model in sars_order] + ["xlnet_features"]
sars_bias_order = [f"{model}_bias_adj" for model in sars_order]
model_order = baseline_order + sars_base_order + sars_full_order + sars_bias_order


def map_model_names(name):
    if name in ["gru", "bert", "xlnet"]:
        return name
    elif name in ["ials", "als"]:
        new_name = "iALS"
    else:
        new_name = name.upper()
    return (new_name, name)


def get_gpu_table(gpu_dict: dict):
    gpu_df = pd.DataFrame.from_dict(gpu_dict, orient="index")

    # gpu_df.index = gpu_df.index.map(map_model_names)
    print(gpu_df)
    # gpu_df = gpu_df.reindex(model_order, level=0)
    return gpu_df


if __name__ == "__main__":
    log_path = Path("logs")
    mooc_log_paths = log_path.glob("*mooc_*eval.log")
    mooc_baseline_log_paths = (log_path / "fix").glob("mooc*baseline*eval_fix.log")
    # print(*mooc_log_paths, sep="\n")
    mooc_gpus = {
        parse_name(path): get_gpu_from_log(path)
        for path in list(mooc_log_paths) + list(mooc_baseline_log_paths)
    }
    ednet_log_paths = log_path.glob("*ednet_*eval.log")
    ednet_baseline_log_paths = (log_path / "fix").glob("ednet*baseline*eval_fix.log")
    ednet_gpus = {
        parse_name(path): get_gpu_from_log(path)
        for path in list(ednet_log_paths) + list(ednet_baseline_log_paths)
    }
    # mooc_gpus_df = get_gpu_table(mooc_gpus)
    mooc_gpu_df = pd.DataFrame.from_dict(mooc_gpus, orient="index")
    ednet_gpu_df = pd.DataFrame.from_dict(ednet_gpus, orient="index")
    gpus_df = pd.concat(
        [ednet_gpu_df, mooc_gpu_df],
        axis=1,
        keys=["EdNet", "MOOCCUBEX"],
        names=["Model"],
    )
    # ednet_gpus_df = get_gpu_table(ednet_gpus)
    print(
        gpus_df.to_latex(
            position="h", caption="GPUs user for evaluations", multicolumn=True
        )
    )
