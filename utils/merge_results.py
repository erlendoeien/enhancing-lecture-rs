from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--paths", "-p", nargs="*", help="Paths to of base result files to merge"
    )
    args = vars(parser.parse_args())
    paths = [Path(path) for path in args["paths"]]
    print(*paths, sep="\n")
    # base_path = Path("results")
    for path in paths:
        if not path.exists():
            raise UserWarning(f"{path}  does not exist")
            continue
        print(f"Merging {path}")
        path_tokens = path.stem.split("_")
        other_name = "_".join(path_tokens[:-1] + ["cont"] + [path_tokens[-1]])
        other_path = path.parent / f"{other_name}.parquet"
        if not other_path.exists():
            raise UserWarning(f"Other path {other_path} does not exist -> Can't merge")
            continue
        main_df = pd.read_parquet(path)
        other_df = pd.read_parquet(other_path)
        merged_stem = f"{'_'.join(path_tokens[:-1] + ['merged'] + [path_tokens[-1]])}"

        merged_path = path.parent / merged_stem
        merged = pd.concat([main_df, other_df])
        merged.to_parquet(path.parent / f"{merged_stem}.parquet")
        merged.to_json(path.parent / f"{merged_stem}.json", indent=4, orient="index")
    print("COMPLETE")
