from dataclasses import dataclass
from typing import List, Optional, Union, Any
import torch
import torch.nn as nn

from lit_gpt.adapter import GPT as BaseAdapter, Config as AdapterConfig, mark_only_adapter_as_trainable
from lit_gpt.lora import GPT as BaseLoRA, Config as LoRAConfig, mark_only_lora_as_trainable


ADAPTER_TYPE = "adapter"
LORA_TYPE = "lora"

@dataclass
class ClassificationConfig(LoRAConfig, AdapterConfig):
    output_size: int = 28
    model_type: str = LORA_TYPE

class ClassificationGPT(BaseAdapter, BaseLoRA):
    
    def __init__(self, config: ClassificationConfig) -> None:
        self.model_type = config.model_type
        adapter_or_lora(
            self.model_type,
            BaseAdapter.__init__,
            BaseLoRA.__init__
        )(self, config)
        
        self.classification_head = nn.Linear(config.n_embd, config.output_size)
    
    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return torch.nn.functional.sigmoid(self.classification_head(x[:, -1, :]))  # (b, output size)
    
    def _init_weights(self, module: nn.Module) -> None:
        adapter_or_lora(
            self.model_type,
            BaseAdapter._init_weights,
            BaseLoRA._init_weights
        )(self, module)
            
    
def mark_trainable(model: ClassificationGPT) -> None:
    adapter_or_lora(
        model.model_type,
        mark_only_adapter_as_trainable,
        mark_only_lora_as_trainable
    )(model)
    model.classification_head.requires_grad_ = True
    

def adapter_or_lora(
    model_type: str,
    adapter_return: Any,
    lora_return: Any,
    adapter_args: dict | None = None,
    lora_args: dict | None = None
) -> Any:
    if model_type == ADAPTER_TYPE:
        if adapter_args is not None:
            return adapter_return(**adapter_args)
        return adapter_return
    elif model_type == LORA_TYPE:
        if lora_args is not None:
            return lora_return(**lora_args)
        return lora_return
    else:
        raise ValueError(f'"model_type" should be "{LORA_TYPE}" or "{ADAPTER_TYPE}", found {model_type}')