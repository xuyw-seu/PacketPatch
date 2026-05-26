import random
import os
import numpy as np
import torch

def set_seed(seed=7):
    """Summary of set_seed.
    
    Args:
        seed (Any): Description.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

