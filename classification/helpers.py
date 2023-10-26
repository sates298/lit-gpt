import torch
import lightning as L
from typing import Optional, Tuple, List, Dict, Union
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
    

def get_batch(
    fabric: L.Fabric, data: List[Dict], micro_batch_size: int, longest_seq_ix: Optional[int] = None, first_ix: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert first_ix == -1 or first_ix >= 0
    if first_ix == -1:
        ix = torch.randint(len(data), (micro_batch_size,))
    else:
        last_ix= first_ix+micro_batch_size 
        if last_ix > len(data):
            last_ix = len(data)
        ix = list(range(first_ix, last_ix))
        
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix
    
    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.float64) for i in ix]
    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack(labels)

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor
) -> torch.Tensor:    
    return torch.nn.functional.cross_entropy(logits, targets)