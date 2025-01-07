import torch
import numpy as np
import torch.nn as nn

def inject_variations(weights, sigma, scale_factor=None, quantized=False):
    """
    Inject variations into the weights.
    - If quantized: Inject noise on quantized weights (INT scale).
    - If not quantized: Inject noise on floating-point weights (FP32).
    """
    lognormal_variations = np.random.lognormal(mean=0, sigma=sigma, size=weights.shape)
    variations = torch.tensor(lognormal_variations, dtype=weights.dtype, device=weights.device)

    if quantized:
        # Variations on INT scale
        noisy_weights = weights + scale_factor * variations
    else:
        # Variations on FP32 scale
        noisy_weights = weights * variations

    return noisy_weights


def apply_variations(model, sigma):
    """
    Apply variations to all Conv2d and Linear layers, excluding the first Conv2D and last FC layer.
    """
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # Exclude the first Conv2D and last FC layers
                if "conv0" in name or "fc" in name:
                    continue

                if hasattr(layer, "quantize_fn"):
                    # Quantized weights: Apply noise on INT scale
                    wbit = getattr(layer, "wbit", 32)
                    scale_factor = layer.weight.abs().max() / (2 ** (wbit - 1) - 1)
                    layer.weight.data = inject_variations(
                        layer.weight.data, sigma=sigma, scale_factor=scale_factor, quantized=True
                    )
                else:
                    # Non-quantized weights: Apply noise on FP32 scale
                    layer.weight.data = inject_variations(layer.weight.data, sigma=sigma, quantized=False)
