from pathlib import Path
from copy import deepcopy
from datetime import datetime
import re
import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def normalize_config(config):
    config = deepcopy(config or {})

    experiment_cfg = config.setdefault("experiment", {})
    experiment_cfg.setdefault("name", "experiment")
    experiment_cfg.setdefault("result_root", "result")
    experiment_cfg.pop("run_prefix", None)
    experiment_cfg.pop("run_dir", None)

    runtime_cfg = config.setdefault("runtime", {})
    if runtime_cfg.get("seed") in ("None", "none", ""):
        runtime_cfg["seed"] = None
    runtime_cfg.setdefault("seed", None)
    runtime_cfg.setdefault("device", "cuda")
    runtime_cfg.setdefault("gpu_ids", None)

    distributed_cfg = runtime_cfg.setdefault("distributed", {})
    distributed_cfg.setdefault("enabled", False)
    distributed_cfg.setdefault("backend", "nccl")
    distributed_cfg.pop("nproc_per_node", None)

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg.setdefault("train", {})
    dataset_cfg.setdefault("validation", {})

    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("name", "model")
    model_cfg.setdefault("args", {})

    trainer_cfg = config.setdefault("trainer", {})
    trainer_cfg.setdefault("resume", None)

    return config


def create_result_dir(config, timestamp=None):
    config = normalize_config(config)
    experiment_cfg = config["experiment"]

    experiment_name = _slugify(experiment_cfg.get("name", "experiment"))
    dataset_name = _slugify(_get_dataset_name(config))
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    result_name = f"result_{experiment_name}_{dataset_name}_{timestamp}"
    result_root = Path(experiment_cfg.get("result_root", "result"))
    result_dir = result_root / result_name

    suffix = 1
    while result_dir.exists():
        result_dir = result_root / f"{result_name}_{suffix:03d}"
        suffix += 1

    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def load_train_config(config_path, resume=None):
    config_path = Path(config_path)
    resume = Path(resume) if resume else None

    if resume and not resume.exists():
        raise FileNotFoundError(f"Resume path does not exist: {resume}")

    if resume and resume.is_dir():
        resume_config_path = resume / "config.yaml"
        if not resume_config_path.exists():
            raise FileNotFoundError(f"Resume config does not exist: {resume_config_path}")

        config = normalize_config(load_config(resume_config_path))
        result_dir = resume
    else:
        config = normalize_config(load_config(config_path))
        result_dir = None

    if resume:
        config.setdefault("trainer", {})["resume"] = str(resume)

    return config, result_dir


def prepare_result_dir(config, result_dir=None):
    from utils.utils_dist import broadcast_object, is_main_process

    if result_dir is None and is_main_process():
        result_dir = create_result_dir(config)

    result_dir = Path(broadcast_object(str(result_dir)))
    config.setdefault("experiment", {})["result_dir"] = str(result_dir)
    config["experiment"].pop("run_dir", None)
    return result_dir


def save_result_config(config, result_dir):
    from utils.utils_dist import barrier, is_main_process

    if is_main_process():
        save_config(config, Path(result_dir) / "config.yaml")
    barrier()


def _get_dataset_name(config):
    dataset_cfg = config.get("dataset", {})

    if isinstance(dataset_cfg.get("name"), str):
        return dataset_cfg["name"]

    train_cfg = dataset_cfg.get("train", {})
    if isinstance(train_cfg, dict):
        if train_cfg.get("name"):
            return train_cfg["name"]
        if train_cfg.get("root"):
            return Path(train_cfg["root"]).name

    return "dataset"


def _slugify(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "unknown"
