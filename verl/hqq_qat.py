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
    update_metadata: str
    max_quantized_value: float
    _scale_ema: Optional[Tensor] = None
    _zero_ema: Optional[Tensor] = None
    beta: float = 0.0
    use_qat: bool = True
    num_updates: int = 0

    @property
    def scale(self) -> Optional[Tensor]:
        if self._scale_ema is None:
            return None

        bias_correction = 1 - self.beta**self.num_updates
        return self._scale_ema / bias_correction

    @property
    def zero(self) -> Optional[Tensor]:
        if self._zero_ema is None:
            return None

        bias_correction = 1 - self.beta**self.num_updates
        return self._zero_ema / bias_correction

    def update(self, new_scale: Tensor, new_zero: Tensor) -> None:
        self.num_updates += 1

        if self._scale_ema is None or self._zero_ema is None:
            self._scale_ema = torch.zeros_like(new_scale)
            self._zero_ema = torch.zeros_like(new_zero)

        self._scale_ema = self.beta * self._scale_ema + (1.0 - self.beta) * new_scale
        self._zero_ema = self.beta * self._zero_ema + (1.0 - self.beta) * new_zero


class FakeHQQLinear(nn.Linear):
    fake_hqq_data: FakeHQQData

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.fake_hqq_data.use_qat:
            if self.fake_hqq_data.scale is None or self.fake_hqq_data.zero is None:
                assert self.fake_hqq_data.update_metadata == "actor"

                with torch.no_grad():
                    _, metadata = Quantizer.quantize(
                        self.weight.detach(),
                        device=input.device,
                        compute_dtype=input.dtype,
                        **self.fake_hqq_data.quant_config,
                    )
                    self.fake_hqq_data.update(metadata["scale"].to(self.weight.dtype), metadata["zero"].to(self.weight.dtype))

            if self.fake_hqq_data.quant_config["axis"] == 1:
                weight = self.weight.view(-1, self.fake_hqq_data.quant_config["group_size"])
            else:
                weight = self.weight.view(self.fake_hqq_data.quant_config["group_size"], -1)

            fake_quantized_weight: torch.Tensor = QuantizeSTE.apply(
                weight,
                self.fake_hqq_data.scale.detach(),
                self.fake_hqq_data.zero.detach(),
                self.fake_hqq_data.max_quantized_value,
            )

            return F.linear(input, fake_quantized_weight.view(self.weight.shape), self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


def patch_linear_with_fake_hqq(
    linear: nn.Linear,
    hqq_qat_config: dict[str, Any],
) -> None:
    linear.fake_hqq_data = FakeHQQData(
        quant_config=hqq_qat_config["quant_config"],
        update_metadata=hqq_qat_config["update_metadata"],
        max_quantized_value=round(2 ** hqq_qat_config["quant_config"]["nbits"] - 1),
        scale=None,
        zero=None,
        beta=hqq_qat_config["beta"],
        use_qat=True,
    )
    linear.forward = FakeHQQLinear.forward.__get__(linear, type(linear))


def replace_linear_with_fake_hqq(
    module: nn.Module,
    hqq_qat_config: dict[str, Any],
) -> None:
    for child in list(module.children()):
        if isinstance(child, nn.Linear):
            patch_linear_with_fake_hqq(child, hqq_qat_config)
        else:
            replace_linear_with_fake_hqq(child, hqq_qat_config)


def clear_fake_hqq_quant_data(module: nn.Module) -> None:
    for child in list(module.children()):
        if isinstance(child, nn.Linear):
            if hasattr(child, "fake_hqq_data"):
                assert isinstance(child.fake_hqq_data, FakeHQQData)
                child.fake_hqq_data.scale = None
                child.fake_hqq_data.zero = None
        else:
            clear_fake_hqq_quant_data(child)


def enable_hqq_qat(module: nn.Module) -> None:
    for child in list(module.children()):
        if isinstance(child, nn.Linear):
            if hasattr(child, "fake_hqq_data"):
                assert isinstance(child.fake_hqq_data, FakeHQQData)
                child.fake_hqq_data.use_qat = True
        else:
            enable_hqq_qat(child)


def disable_hqq_qat(module: nn.Module) -> None:
    for child in list(module.children()):
        if isinstance(child, nn.Linear):
            if hasattr(child, "fake_hqq_data"):
                assert isinstance(child.fake_hqq_data, FakeHQQData)
                child.fake_hqq_data.use_qat = False
        else:
            disable_hqq_qat(child)
