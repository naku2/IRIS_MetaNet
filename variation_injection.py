import torch
import numpy as np
import torch.nn as nn

# def inject_variations(weights, sigma, scale_factor=None, quantized=False):
#     """
#     Inject variations into the weights.
#     - If quantized: Inject noise on quantized weights (INT scale).
#     - If not quantized: Skip variation injection.
#     """
#     if not quantized:
#         # Non-quantized weights: Skip variation injection
#         return weights

#     # Generate lognormal variations
#     lognormal_variations = np.random.lognormal(mean=0, sigma=sigma, size=weights.shape)
#     variations = torch.tensor(lognormal_variations, dtype=weights.dtype, device=weights.device)

#     # Variations on INT scale
#     # Instead of addition, use multiplication to reflect realistic variations
#     noisy_weights = weights * (1 + scale_factor * (variations - 1))
#     return noisy_weights


# def apply_variations(model, sigma):
#     """
#     Apply variations to all Conv2d and Linear layers, excluding non-quantized weights.
#     """
#     with torch.no_grad():
#         for name, layer in model.named_modules():
#             if isinstance(layer, (nn.Conv2d, nn.Linear)):
#                 # Check if the layer is quantized
#                 is_quantized = hasattr(layer, "quantize_fn") and (hasattr(layer, "wbit") and layer.wbit != 32)

#                 if is_quantized:
#                     # Calculate scale factor for INT scale variations
#                     if layer.wbit == 1:
#                         scale_factor = layer.weight.abs().max()
#                     else:
#                         scale_factor = layer.weight.abs().max() / (2 ** (layer.wbit - 1) - 1)
#                     layer.weight.data = inject_variations(
#                         layer.weight.data, sigma=sigma, scale_factor=scale_factor, quantized=True
#                     )
#                 else:
#                     # Skip non-quantized weights
#                     print(f"Skipping variation for non-quantized weights in layer: {name}")

def inject_variations(weights, sigma, scale_factor=None):
    """
    Inject variations into the weights (tensor level).
    """
    # ✅ Log-normal variations 생성
    lognormal_variations = np.random.lognormal(mean=0, sigma=sigma, size=weights.shape)
    variations = torch.tensor(lognormal_variations, dtype=weights.dtype, device=weights.device)

    # ✅ 변동량을 곱하여 variation 주입
    noisy_weights = weights * (1 + scale_factor * (variations - 1))
    return noisy_weights


def apply_variations(weights, sigma, wbit):
    """
    Apply variations to quantized weights at tensor level.
    기존 모델 코드와 일치하는 조건 적용.
    """
    if not isinstance(weights, torch.Tensor):
        raise ValueError("apply_variations expects a tensor input, but got {}".format(type(weights)))

    # # ✅ 기존 모델 코드와 동일한 조건 반영
    # if wbit == 1:
    #     scale_factor = weights.abs().max()
    # else:
    #     scale_factor = weights.abs().max() / (2 ** (wbit - 1) - 1)

    if wbit != 32:
        # ✅ uint8 → float 변환 없이 평균 계산
        max_n = 2**wbit - 1  # MAX_N 값 (양자화 범위 최대값)
        e_abs_w = weights.to(dtype=torch.int32).abs().sum() // weights.numel()  # 정수 연산을 활용한 평균 계산

        scale_factor = e_abs_w / max_n
    
    weights = inject_variations(weights, sigma=sigma, scale_factor=scale_factor)

    return weights