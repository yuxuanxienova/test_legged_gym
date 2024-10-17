import torch
import numpy as np
import matplotlib.pyplot as plt
def my_pattern_func(pattern_cfg, device):
    # Example: generate rays in a circular pattern
    num_rays = 16
    angles = torch.linspace(0, 2 * torch.pi, num_rays, device=device)
    ray_directions = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts.unsqueeze(0), ray_directions.unsqueeze(0)