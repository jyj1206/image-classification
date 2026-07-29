from models import CIFAR10FilterNet, CIFAR10PlainFilterNet


def build_model(config):
    model_config = config["model"]
    models = {
        "cifar10_filter_net": CIFAR10FilterNet,
        "cifar10_plain_filter_net": CIFAR10PlainFilterNet,
    }
    name = model_config["name"].lower()
    if name not in models:
        raise ValueError(f"Unsupported model: {model_config['name']}")
    return models[name](**model_config.get("args", {}))
