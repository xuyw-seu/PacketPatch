import torch


def save_model(model, model_path):
    """Summary of save_model.
    
    Args:
        model (Any): Description.
        model_path (Any): Description.
    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
