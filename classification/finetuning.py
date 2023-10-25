import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from tqdm import tqdm

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
from classification.helpers import setup_quantization, main_lora_config,\
    get_batch, cross_entropy
from classification.model import ClassificationGPT, ClassificationConfig,\
    mark_trainable, ADAPTER_TYPE, LORA_TYPE, adapter_or_lora
from finetune.full import get_longest_seq_length
from finetune.adapter import save_adapter_checkpoint, hparams as adapter_hparams
from finetune.lora import save_lora_checkpoint, hparams as lora_hparams
from lit_gpt.adapter import Block as AdapterBlock
from lit_gpt.lora import Block as LoRABlock 
# from lit_gpt.adapter import GPT, Block, Config, adapter_filter, mark_only_adapter_as_trainable
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    num_parameters,
    load_checkpoint,
)

custom_hparams = dict(
    output_size=28,
    with_scheduler=True,
    micro_batch_size=2,
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
    hparams["model_type"] = model_type
    hparams["quantize"] = quantize
    hparams.update(custom_hparams)
    
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

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

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
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams["max_iters"] // hparams["batch_size"])
    
    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, scheduler, train_data, val_data, checkpoint_dir, out_dir, speed_monitor, hparams)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = out_dir / f"lit_model_{model.model_type}_finetuned.pth"
    
    if model.model_type == ADAPTER_TYPE:
        save_adapter_checkpoint(fabric, model, save_path)
    elif model.model_type == LORA_TYPE:
        save_lora_checkpoint(fabric, model, save_path)
    
    test(fabric, model, test_data, checkpoint_dir)


def train(
    fabric: L.Fabric,
    model: ClassificationGPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    hparams: dict,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = longest_seq_length
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, val_data, tokenizer, hparams["eval_iters"], hparams["micro_batch_size"], sanity_check=True)  # sanity check

    with torch.device("meta"):
        meta_model = ClassificationGPT(model.config)
        mark_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * hparams["micro_batch_size"]
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (hparams["micro_batch_size"], longest_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num in tqdm(range(hparams["max_iters"]), desc="Training"):
        if step_count <= hparams["warmup_steps"]:
            # linear warmup
            lr = hparams["learning_rate"] * step_count / hparams["warmup_steps"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(fabric, train_data, hparams["micro_batch_size"], longest_seq_ix if iter_num == 0 else None)
        
        is_accumulating = (iter_num + 1) % hparams["gradient_accumulation_iters"] != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = cross_entropy(logits, targets)
            fabric.backward(loss / hparams["gradient_accumulation_iters"])
            # TODO add metrics
            # TODO add logging
            
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and step_count > hparams["warmup_steps"]:
                scheduler.step()
            step_count += 1



        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * hparams["micro_batch_size"],
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        # if iter_num % hparams["log_interval"] == 0:
        #     fabric.print(
        #         f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
        #         f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
        #     )
        if not is_accumulating and step_count % hparams["eval_interval"] == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, hparams["eval_iters"], hparams["micro_batch_size"])
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            # fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % hparams["save_interval"] == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            adapter_or_lora(
                model.model_type,
                save_adapter_checkpoint,
                save_lora_checkpoint
            )(fabric, model, checkpoint_path)
                

# the adapter "kv cache" cannot be initialized under `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: ClassificationGPT, 
             val_data: List[Dict], tokenizer: Tokenizer,
             eval_iters: int,
             micro_batch_size: int,
             sanity_check: bool = False) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters)):
        input_ids, targets = get_batch(fabric, val_data, micro_batch_size=micro_batch_size)
        logits = model(input_ids)
        losses[k] = cross_entropy(logits, targets)
        if not sanity_check:
            # TODO add metrices
            # TODO add logging
            pass
    val_loss = losses.mean()

    model.train()
    return val_loss

@torch.no_grad()
def test(fabric: L.Fabric, model: ClassificationGPT, val_data: List[Dict], checkpoint_dir: Path):
    tokenizer = Tokenizer(checkpoint_dir)
    fabric.print("Testing ...")
    model.eval()
    

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
