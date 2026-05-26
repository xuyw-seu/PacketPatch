# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


def count_lines(file_path):
    """Summary of count_lines.
    
    Args:
        file_path (Any): Description.
    Returns:
        Any: Description.
    """
    lines_num = 0
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(2^20)
            if not data:
                break
            lines_num += data.count(b'\n')
    return lines_num


def flip(x, dim):
    """Summary of flip.
    
    Args:
        x (Any): Description.
        dim (Any): Description.
    Returns:
        Any: Description.
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]
