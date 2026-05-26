import torch


def load_model(model, model_path):
    """Summary of load_model.
    
    Args:
        model (Any): Description.
        model_path (Any): Description.
    Returns:
        Any: Description.
    """
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model
