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
            self.fake_hqq_data.quant_data["scale"].detach_(),
            self.fake_hqq_data.quant_data["zero"].detach_(),
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


def update_hf_metadata_from_vllm(hf_model: nn.Module, vllm_model: nn.Module) -> None:
    metadata = {}
    for name, _ in vllm_model.named_parameters():
        if name[-len(".W_q") :] == ".W_q":
            quant_layer_name = name[: -len(".W_q")]
            quant_layer = vllm_model.get_submodule(quant_layer_name)
            scale = quant_layer.meta["scale"]
            zero = quant_layer.meta["zero"]

            if "qkv_proj" in quant_layer_name:
                qkv_linear_layer = vllm_model.get_submodule(quant_layer_name[: -len(".quant_layer")])
                num_heads = qkv_linear_layer.num_heads
                num_kv_heads = qkv_linear_layer.num_kv_heads
                num_total_heads = num_heads + 2 * num_kv_heads

                n = scale.shape[0] // num_total_heads
                assert scale.shape[0] == num_total_heads * n and zero.shape[0] == num_total_heads * n

                q_proj_scale = scale[: n * num_heads]
                k_proj_scale = scale[n * num_heads : n * (num_heads + num_kv_heads)]
                v_proj_scale = scale[n * (num_heads + num_kv_heads) :]

                q_proj_zero = zero[: n * num_heads]
                k_proj_zero = zero[n * num_heads : n * (num_heads + num_kv_heads)]
                v_proj_zero = zero[n * (num_heads + num_kv_heads) :]

                metadata[quant_layer_name.replace("qkv_proj", "q_proj").replace(".quant_layer", "")] = {
                    "scale": q_proj_scale,
                    "zero": q_proj_zero,
                }
                metadata[quant_layer_name.replace("qkv_proj", "k_proj").replace(".quant_layer", "")] = {
                    "scale": k_proj_scale,
                    "zero": k_proj_zero,
                }
                metadata[quant_layer_name.replace("qkv_proj", "v_proj").replace(".quant_layer", "")] = {
                    "scale": v_proj_scale,
                    "zero": v_proj_zero,
                }
            elif "gate_up_proj" in quant_layer_name:
                n = scale.shape[0] // 2
                assert 2 * n == scale.shape[0] and 2 * n == zero.shape[0]

                gate_proj_scale = scale[:n]
                up_proj_scale = scale[n:]

                gate_proj_zero = zero[:n]
                up_proj_zero = zero[n:]

                metadata[quant_layer_name.replace("gate_up_proj", "gate_proj").replace(".quant_layer", "")] = {
                    "scale": gate_proj_scale,
                    "zero": gate_proj_zero,
                }
                metadata[quant_layer_name.replace("gate_up_proj", "up_proj").replace(".quant_layer", "")] = {
                    "scale": up_proj_scale,
                    "zero": up_proj_zero,
                }
            else:
                metadata[quant_layer_name.replace(".quant_layer", "")] = {"scale": scale, "zero": zero}

    for name, quant_data in metadata.items():
        linear = hf_model.get_submodule(name)
        assert hasattr(linear, "fake_hqq_data")
        assert isinstance(linear.fake_hqq_data, FakeHQQData)
        linear.fake_hqq_data.quant_data = quant_data
