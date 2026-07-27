from pathlib import Path


def prepare_checkpoint_dir(result_dir, config=None):
    checkpoint_dir = Path(result_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if config is not None:
        config.setdefault("trainer", {})["checkpoint_dir"] = str(checkpoint_dir)

    return checkpoint_dir


def get_checkpoint_path(result_dir, name):
    return Path(result_dir) / "checkpoints" / name


def get_latest_checkpoint_path(result_dir):
    return get_checkpoint_path(result_dir, "latest.pth")


def get_best_checkpoint_path(result_dir):
    return get_checkpoint_path(result_dir, "best.pth")
