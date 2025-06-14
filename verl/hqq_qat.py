from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hqq.core.quantize import Quantizer
from torch import Tensor


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        scale: Tensor,
        zero: Tensor,
        max_quantized_value: int,
    ) -> Tensor:
        x_quantized = (x / scale + zero).round().clamp(0, max_quantized_value)
        return (x_quantized - zero) * scale

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[Tensor, None, None, None]:  # type: ignore
        return grad_out, None, None, None


class FakeHQQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, quant_config: Optional[dict] = None) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.quant_config = {
            "nbits": 4,
            "group_size": 64,
            "axis": 1,
            "round_zero": True,
            "optimize": True,
            "bitpack": False,
        }
        if quant_config:
            self.quant_config.update(quant_config)

        self.max_quantized_value = round(2 ** self.quant_config["nbits"] - 1)

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, meta = Quantizer.quantize(
                x,
                device=x.device,  # type: ignore
                compute_dtype=x.dtype,
                **self.quant_config,
            )

        input_shape = x.shape
        if self.quant_config["axis"] == 1:
            x = x.view(-1, self.quant_config["group_size"])
        else:
            x = x.view(self.quant_config["group_size"], -1)

        return QuantizeSTE.apply(
            x,
            meta["scale"].detach_(),
            meta["zero"].detach_(),
            self.max_quantized_value,
        ).view(input_shape)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.fake_quantize(self.weight), self.bias)


def replace_linear_with_fake_hqq(
    module: nn.Module,
    quant_config: Optional[dict] = None,
) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            fake = FakeHQQLinear(child, quant_config)
            setattr(module, name, fake)
        else:
            replace_linear_with_fake_hqq(child, quant_config)

    return module
