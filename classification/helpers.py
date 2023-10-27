import torch
import lightning as L
from typing import Optional, Tuple
from lightning.fabric.plugins import BitsandbytesPrecision

from classification.model import ClassificationConfig

def setup_quantization(quantize: Optional[str],
                       precision: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None
    return plugins, precision

def main_lora_config(name: str, hparams: dict) -> ClassificationConfig:
    return ClassificationConfig.from_name(
        name=name,
        r=hparams["lora_r"],
        alpha=hparams["lora_alpha"],
        dropout=hparams["lora_dropout"],
        to_query=hparams["lora_query"],
        to_key=hparams["lora_key"],
        to_value=hparams["lora_value"],
        to_projection=hparams["lora_projection"],
        to_mlp=hparams["lora_mlp"],
        to_head=hparams["lora_head"],
    )

