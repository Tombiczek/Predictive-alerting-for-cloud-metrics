import torch.nn as nn
from tsai.models.InceptionTime import InceptionTime


MODELS: dict[str, type[nn.Module]] = {
    "InceptionTime": InceptionTime,
}


def build_model(config: dict) -> nn.Module:
    """
    Build a time series classification model from config.

    Args:
        config: Configuration dict containing:
            - model_name: Name of the model
            - window_size: Length of input sequences
            - model_kwargs: Optional dict of model-specific parameters

    Returns:
        An nn.Module ready for training
    """
    model_name = config.get("model_name", "InceptionTime")
    model_cls = MODELS[model_name]
    model_kwargs = config.get("model_kwargs", {})
    # tsai models expect: (c_in, c_out, seq_len, **kwargs)
    c_in = 1
    c_out = 1
    seq_len = config["window_size"]

    return model_cls(c_in=c_in, c_out=c_out, seq_len=seq_len, **model_kwargs)

