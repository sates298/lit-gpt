import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from tqdm import tqdm
from torch.utils.data import DataLoader

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.plugins import BitsandbytesPrecision


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from generate.base import generate
from classification.metrics import ClassificationMetrics
from classification.dataloader import PaddingBatcher
from classification.helpers import setup_quantization, main_lora_config
from classification.model import ClassificationGPT, ClassificationConfig,\
    mark_trainable, ADAPTER_TYPE, LORA_TYPE, adapter_or_lora
from classification.training import train, valid_test

from finetune.adapter import hparams as adapter_hparams
from finetune.lora import hparams as lora_hparams
from lit_gpt.adapter import Block as AdapterBlock
from lit_gpt.lora import Block as LoRABlock 

from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    num_parameters,
    load_checkpoint,
)

custom_hparams = dict(
    output_size=28,
    with_scheduler=True,
    batch_size=128,
    micro_batch_size=8, # needs to be a divisor of the batch_size
    devices=1,
    max_epochs=10,
    patience= 3,
    epsilon=0.01
)

def setup(
    data_dir: Path = Path("data/csv/goemotions"),
    checkpoint_dir: Path = Path("checkpoints/mistralai/Mistral-7B-Instruct-v0.1"),
    out_dir: Path = Path("out/finetuning/goemotions"),
    precision: Optional[str] = None,
    model_type: str = ADAPTER_TYPE,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    **kwargs
) -> None:
    precision = precision or get_default_supported_precision(training=True)

    global custom_hparams
    custom_hparams |= kwargs

    hparams = adapter_or_lora(model_type, adapter_hparams, lora_hparams)
    hparams.update(custom_hparams)
    hparams["model_type"] = model_type
    hparams["quantize"] = quantize
    hparams["gradient_accumulation_iters"] = hparams["batch_size"] // hparams["micro_batch_size"]
    assert hparams["gradient_accumulation_iters"] > 0
    
    plugins, precision = adapter_or_lora(model_type, (None, precision), setup_quantization, lora_args={"quantize": quantize, "precision": precision})
    
    fabric_devices = hparams["devices"]
    block_cls = adapter_or_lora(model_type, AdapterBlock, LoRABlock)
    if fabric_devices > 1:
        if quantize and model_type == LORA_TYPE:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantization flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={block_cls},
            activation_checkpointing_policy={block_cls},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=hparams["log_interval"])
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, hparams)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, hparams: dict) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "val.pt")
    test_data = torch.load(data_dir / "test.pt")

    config = adapter_or_lora(
        hparams["model_type"],
        ClassificationConfig.from_name,
        main_lora_config,
        adapter_args=dict(name=checkpoint_dir.name),
        lora_args=dict(name=checkpoint_dir.name, hparams=hparams)
    )
    
    config.model_type = hparams["model_type"]
    
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = ClassificationGPT(config)

    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    
    mark_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = None
    if hparams["with_scheduler"]:
        T_max = (hparams["max_epochs"] * len(train_data)) // (hparams["batch_size"]*hparams["micro_batch_size"])
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    fabric.seed_everything(1337 + fabric.global_rank)

    train(fabric, model, optimizer, scheduler, train_data, val_data, out_dir, hparams)
    
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    
    test_loader = PaddingBatcher(fabric, test_data, hparams["micro_batch_size"]).dataloader()
    valid_test(fabric, model, test_loader, ClassificationMetrics(model.config.output_size), mode="test")




if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
