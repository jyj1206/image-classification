from models import CIFAR10FilterNet


def build_model(config):
    model_config = config["model"]
    if model_config["name"].lower() != "cifar10_filter_net":
        raise ValueError(f"Unsupported model: {model_config['name']}")
    return CIFAR10FilterNet(**model_config.get("args", {}))
