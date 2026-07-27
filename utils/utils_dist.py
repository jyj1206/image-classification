import os

import torch
import torch.distributed as dist


def configure_visible_devices(gpu_ids=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]

    if not gpu_ids:
        return None

    visible_devices = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return visible_devices


def is_torchrun_launched():
    return all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def get_runtime_config(config):
    return config.get("runtime", {}) if isinstance(config, dict) else {}


def get_distributed_config(config):
    runtime_cfg = get_runtime_config(config)
    return runtime_cfg.get("distributed", {}) if isinstance(runtime_cfg, dict) else {}


def init_distributed(config=None):
    runtime_cfg = get_runtime_config(config)
    dist_cfg = get_distributed_config(config)

    device_name = runtime_cfg.get("device", "cuda")
    gpu_ids = runtime_cfg.get("gpu_ids", None)
    configure_visible_devices(gpu_ids)

    dist_enabled = dist_cfg.get("enabled", False)
    backend = dist_cfg.get("backend", "auto")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    distributed = bool(dist_enabled) or world_size > 1

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not distributed:
        return {
            "distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": device,
            "gpu_ids": gpu_ids,
        }

    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    return {
        "distributed": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "gpu_ids": gpu_ids,
    }


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def broadcast_object(value, src=0):
    if not is_distributed():
        return value

    objects = [value]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def reduce_mean_tensor(tensor):
    if not is_distributed():
        return tensor

    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def reduce_sum_tensor(tensor):
    if not is_distributed():
        return tensor

    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model
