from torch import nn


def build_loss(config):
    name = config["trainer"].get("loss", "cross_entropy").lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {name}")
