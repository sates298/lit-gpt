import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
import lightning as L

class PaddingBatcher():
    
    def __init__(self, fabric: L.Fabric, data: List[Dict], batch_size: int) -> None:
        self.fabric = fabric 
        self.data = data
        self.batch_size = batch_size
        
    def dataloader(self, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.data, batch_size=self.batch_size, collate_fn=self.collate_fn_padding, shuffle=shuffle)

    def collate_fn_padding(self, data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = [item["input_ids"].type(torch.int64) for item in data]
        labels = [item["labels"].type(torch.float64) for item in data]
        # this could be `longest_seq_length` to have a fixed size for all batches
        max_len = max(len(s) for s in input_ids)

        def pad_right(x, pad_id):
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

        x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
        y = torch.stack(labels)

        if self.fabric.device.type == "cuda" and x.device.type == "cpu":
            x, y = self.fabric.to_device((x.pin_memory(), y.pin_memory()))
        else:
            x, y = self.fabric.to_device((x, y))
        return x, y

    