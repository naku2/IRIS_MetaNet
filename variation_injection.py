import torch
import numpy as np
import torch.nn as nn

def inject_variations(weights, sigma):
    """
    Inject lognormal variations into the weights.
    """
    lognormal_variations = np.random.lognormal(mean=0, sigma=sigma, size=weights.shape)
    return weights * torch.tensor(lognormal_variations, dtype=weights.dtype, device=weights.device)

def apply_variations(model, sigma):
    """
    Apply variations to all Conv2d and Linear layers in the model.
    """
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = inject_variations(layer.weight.data, sigma=sigma)
