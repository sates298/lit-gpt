import torch
import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union, List, Dict
from pathlib import Path

from finetune.full import get_longest_seq_length
from lit_gpt.speed_monitor import estimate_flops, measure_flops, SpeedMonitorFabric as SpeedMonitor
from finetune.adapter import save_adapter_checkpoint
from finetune.lora import save_lora_checkpoint

from classification.model import ClassificationGPT, adapter_or_lora, mark_trainable
from classification.metrics import ClassificationMetrics
from classification.dataloader import PaddingBatcher


def train(
    fabric: L.Fabric,
    model: ClassificationGPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_data: List[Dict],
    val_data: List[Dict],
    out_dir: Path,
    hparams: dict
) -> None:
    patience = hparams["patience"]
    epsilon = hparams["epsilon"]
    
    train_metric_class = ClassificationMetrics(model.config.output_size)
    val_metric_class = ClassificationMetrics(model.config.output_size)
    
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = longest_seq_length
    
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )
    
    train_loader = PaddingBatcher(fabric, train_data, hparams["micro_batch_size"]).dataloader(shuffle=True)
    val_loader = PaddingBatcher(fabric, val_data, hparams["micro_batch_size"]).dataloader(shuffle=False)
    
    # sanity check
    valid_test(fabric, model, val_loader, val_metric_class, sanity_check=True, mode="valid")
    
    train_flops_measurement(fabric, model, hparams, longest_seq_length)
    
    best_checkpoint_metric = float("inf")
    patience_counter = 0
    step_count = 0
    
    fabric.print("Training...")

    for epoch in range(hparams["max_epochs"]):
        for iter_num, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)):
            scheduler_warmup(step_count, hparams, scheduler, optimizer)

            input_ids, targets = batch
            
            step_count = train_step(fabric, model, input_ids, targets,
                                    hparams, train_metric_class, iter_num,
                                    optimizer, scheduler, step_count)
        
        logging_metrics(fabric, train_metric_class, epoch, step_count, "train")
    
        early_stopping_metric = valid_test(fabric, model, val_loader, val_metric_class, mode="valid", epoch=epoch)
        fabric.barrier()
        
        end_training, patience_counter, best_checkpoint_metric = early_stopping(
            best_checkpoint_metric, early_stopping_metric,
            patience_counter, patience, fabric, model,
            out_dir, epsilon)
        
        if end_training:
            return None
        

def scheduler_warmup(step_count: int, hparams: Dict, scheduler, optimizer):
    if step_count <= hparams["warmup_steps"] and scheduler is not None:
        # linear warmup
        lr = hparams["learning_rate"] * step_count / hparams["warmup_steps"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def train_step(fabric: L.Fabric, model: ClassificationGPT,
               input_ids: torch.Tensor, targets: torch.Tensor,
               hparams: Dict, train_metric_class: ClassificationMetrics,
               iter_num: int, optimizer, scheduler, step_count: int):
    is_accumulating = ((iter_num + 1) % hparams["gradient_accumulation_iters"] != 0)
    with fabric.no_backward_sync(model, enabled=is_accumulating):
        logits = model(input_ids)
        loss = loss_function(logits, targets)
        train_metric_class.update(logits, targets)
        fabric.backward(loss / hparams["gradient_accumulation_iters"])
        
    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None and step_count > hparams["warmup_steps"]:
            scheduler.step()
        step_count += 1
    return step_count


def logging_metrics(fabric: L.Fabric, metric_class: ClassificationMetrics,
                    epoch: int, step_count: int, mode: str = "train"):
    metrics = metric_class.compute()
    metric_class.reset()
    metrics["mode"] = mode
    metrics["epoch"] = epoch
    fabric.log_dict(metrics, step=step_count)
    
    
def checkpoint(fabric: L.Fabric, model: ClassificationGPT, out_dir: Path):
    checkpoint_path = out_dir / f"checkpoint-{model.model_type}.pth"
    adapter_or_lora(
        model.model_type,
        save_adapter_checkpoint,
        save_lora_checkpoint
    )(fabric, model, checkpoint_path)   
    
def early_stopping(best_checkpoint_metric: float,
                   early_stopping_metric: float,
                   patience_counter: int,
                   patience: int,
                   fabric: L.Fabric, model: ClassificationGPT,
                   out_dir: Path, epsilon: float):
    
    patience_counter += 1
    if early_stopping_metric + epsilon < best_checkpoint_metric:
        checkpoint(fabric, model, out_dir)
        best_checkpoint_metric = early_stopping_metric
        patience_counter = 0 
    
    end_training = True if patience_counter > patience else False
        
    return end_training, patience_counter, best_checkpoint_metric 

# the adapter "kv cache" cannot be initialized under `inference_mode`
@torch.no_grad()
def valid_test(fabric: L.Fabric, model: ClassificationGPT, data: DataLoader,
               metric_class: ClassificationMetrics, mode: str = "test",
               sanity_check: bool = False, epoch: int = -1):
    introduction = "Testing" if mode == "test" else "Validating"
    model.eval()
    
    sanity_check_iters = 2
    
    losses = []
    for i, batch in tqdm(enumerate(data), total=len(data), desc=introduction):
        if sanity_check and i >= sanity_check_iters:
            break
        input_ids, targets = batch
        logits = model(input_ids)
        metric_class.update(logits, targets)
        losses.append(loss_function(logits, targets))

    if not sanity_check:
        metrics = metric_class.compute()
        metric_class.reset()
        
        metrics["mode"] = mode
        metrics["loss"] = torch.mean(torch.tensor(losses))
        if epoch >= 0:
            metrics["epoch"] = epoch
        fabric.log_dict(metrics)   

    model.train()
    if sanity_check:
        return None
    return metrics["loss"]


def loss_function(
    logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor
) -> torch.Tensor:    
    return torch.nn.functional.cross_entropy(logits, targets)


def train_flops_measurement(fabric, model: ClassificationGPT, hparams: Dict, longest_seq_length: int):
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
    return measured_flops