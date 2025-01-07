import torch
import numpy as np
import torch.nn as nn

def inject_variations(weights, sigma, scale_factor):
    """
    Inject lognormal variations into quantized weights.
    Variations are scaled by the quantization scale factor.
    """
    lognormal_variations = np.random.lognormal(mean=0, sigma=sigma, size=weights.shape)
    variations = torch.tensor(lognormal_variations, dtype=weights.dtype, device=weights.device)
    
    # Apply variations scaled by quantization factor
    modified_weights = weights + scale_factor * variations
    return modified_weights

def apply_variations(model, sigma):
    """
    Apply variations to all quantized Conv2d and Linear layers, excluding the first and last layers.
    """
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Exclude the first Conv2d and last Linear layers
                if "conv0" in name or "fc" in name:
                    continue
                
                # Assume quantization scale factor is layer-specific
                scale_factor = layer.weight.abs().max() / (2**(layer.wbit - 1) - 1)  # Example scale calculation

                # Inject variations into the weights
                layer.weight.data = inject_variations(layer.weight.data, sigma=sigma, scale_factor=scale_factor)
