from dataclasses import dataclass
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
    def backward(ctx: Any, grad_out: Tensor) -> Tuple[Tensor, None, None, None]:
        return grad_out, None, None, None


@dataclass
class FakeHQQData:
    quant_config: dict[str, Any]
    max_quantized_value: float
    quant_data: Optional[dict[str, Any]]


class FakeHQQLinear(nn.Linear):
    fake_hqq_data: FakeHQQData

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.fake_hqq_data.quant_data is None:
            with torch.no_grad():
                _, self.fake_hqq_data.quant_data = Quantizer.quantize(
                    self.weight.detach(),
                    device=input.device,
                    compute_dtype=input.dtype,
                    **self.fake_hqq_data.quant_config,
                )
                for key in self.fake_hqq_data.quant_data:
                    value = self.fake_hqq_data.quant_data[key]
                    if isinstance(value, torch.Tensor):
                        self.fake_hqq_data.quant_data[key] = value.to(self.weight.dtype)

            assert self.fake_hqq_data.quant_data is not None

        if self.fake_hqq_data.quant_config["axis"] == 1:
            weight = self.weight.view(-1, self.fake_hqq_data.quant_config["group_size"])
        else:
            weight = self.weight.view(self.fake_hqq_data.quant_config["group_size"], -1)

        fake_quantized_weight: torch.Tensor = QuantizeSTE.apply(
            weight,
            self.fake_hqq_data.quant_data["scale"].detach(),
            self.fake_hqq_data.quant_data["zero"].detach(),
            self.fake_hqq_data.max_quantized_value,
        )

        return F.linear(input, fake_quantized_weight.view(self.weight.shape), self.bias)


def patch_linear_with_fake_hqq(linear: nn.Linear, quant_config: dict[str, Any]) -> None:
    linear.fake_hqq_data = FakeHQQData(
        quant_config=quant_config,
        max_quantized_value=round(2 ** quant_config["nbits"] - 1),
        quant_data=None,
    )
    linear.forward = FakeHQQLinear.forward.__get__(linear, type(linear))


def replace_linear_with_fake_hqq(module: nn.Module, quant_config: dict[str, Any]) -> None:
    for child in list(module.children()):
        if isinstance(child, nn.Linear):
            patch_linear_with_fake_hqq(child, quant_config)
        else:
            replace_linear_with_fake_hqq(child, quant_config)


def clear_fake_hqq_quant_data(module: nn.Module) -> None:
    for child in list(module.children()):
        if isinstance(child, nn.Linear):
            if hasattr(child, "fake_hqq_data"):
                assert isinstance(child.fake_hqq_data, FakeHQQData)
                child.fake_hqq_data.quant_data = None
        else:
            clear_fake_hqq_quant_data(child)
