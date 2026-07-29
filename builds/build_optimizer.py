import torch


def build_optimizer(config, model):
    cfg = config["trainer"]["optimizer"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = cfg["name"].lower()
    if name == "sgd":
        allowed = {"lr", "momentum", "dampening", "weight_decay", "nesterov"}
        kwargs = {key: value for key, value in cfg.items() if key in allowed}
        return torch.optim.SGD(parameters, **kwargs)
    if name == "adam":
        allowed = {"lr", "betas", "eps", "weight_decay", "amsgrad"}
        kwargs = {key: value for key, value in cfg.items() if key in allowed}
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        allowed = {"lr", "betas", "eps", "weight_decay", "amsgrad"}
        kwargs = {key: value for key, value in cfg.items() if key in allowed}
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {cfg['name']}")


def build_scheduler(config, optimizer):
    cfg = config["trainer"]["scheduler"]
    scheduler_name = cfg["name"].lower()
    if scheduler_name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["trainer"]["epochs"]),
            eta_min=float(cfg.get("eta_min", 1e-5)),
        )
    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler: {cfg['name']}")

    total_epochs = int(config["trainer"]["epochs"])
    warmup_epochs = int(cfg["warmup_epochs"])
    if not 0 < warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be between 1 and epochs - 1")

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(cfg.get("warmup_start_factor", 0.1)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=float(cfg.get("eta_min", 1e-5)),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
