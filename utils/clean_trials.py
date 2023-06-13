import argparse
import logging
from pathlib import Path

from utils.io import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Util for cleaning up trials' model checkpoints")
    parser.add_argument(
        "--trial-dir",
        "-t",
        type=str,
        help="The name of the directory containing the relevant trials and checkpoints",
    )
    args = vars(parser.parse_args())
    trial_path = Path(args["trial_dir"])
    setup_logging(f"logs/clean_{trial_path.name}.log")
    for trial in trial_path.iterdir():
        # Skipping sampler
        if not trial.is_dir():
            continue
        logging.info(f"Loading trial {trial.name}")
        sorted_trials = sorted(
            trial.iterdir(), key=lambda path: -int(path.name.split("-")[-1])
        )
        if len(sorted_trials) == 0:
            trial.rmdir()
            continue
        # latest_chkpt = sorted_trials[0]
        # logging.info(f"Checking {latest_chkpt}")
        # trainer_state_path = latest_chkpt / "trainer_state.json"
        # if not trainer_state_path.exists():
        #     logging.warning(f"Checkpoint {latest_chkpt} do not have a trainer_state")
        #     continue
        # with open(trainer_state_path) as f:
        #     trainer_state = json.load(f)
        # best_checkpoint = Path(trainer_state["best_model_checkpoint"])
        # logging.info(f"Best checkpoint for trial {trial.name}: {best_checkpoint.name}")
        for chkpt in trial.iterdir():
            # if chkpt.name != best_checkpoint.name:
            for file in chkpt.iterdir():
                if file.stem != "trainer_state":
                    file.unlink()
            # logging.info(f"\tDeleting {chkpt}")
            # shutil.rmtree(chkpt)
