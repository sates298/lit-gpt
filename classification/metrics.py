from torchmetrics import Accuracy, F1Score, Recall, Precision
from typing import Any
import torch


class ClassificationMetrics():
    def __init__(self, class_num: int, device: str | int = "cuda") -> None:
        self.device = device
        self.class_num = class_num
        self.accuracy = Accuracy(task="multilabel", num_labels=self.class_num).to(self.device)
        self.f1score = F1Score(task="multilabel", num_labels=self.class_num, average="macro").to(self.device)
        self.recall = Recall(task="multilabel", num_labels=self.class_num, average="macro").to(self.device)
        self.precision = Precision(task="multilabel", num_labels=self.class_num, average="macro").to(self.device)
        
        self.all = {
            "accuracy": self.accuracy,
            "f1-score": self.f1score,
            "recall": self.recall,
            "precision": self.precision
        }
    
    def for_all_metrics(self, function: Any, *args) -> Any:
        return {k: function(metric, *args) for k, metric in self.all.items()}
        
    def _single_update(self, metric, preds, targets):
        return metric.update(preds, targets)
        
    def _single_compute(self, metric):
        return metric.compute()
    
    def _single_reset(self, metric):
        return metric.reset()
    
    def update(self, predictions, targets):
        return self.for_all_metrics(self._single_update, predictions, targets)
        
    def compute(self):
        return self.for_all_metrics(self._single_compute)
        
    def reset(self):
        return self.for_all_metrics(self._single_reset)
