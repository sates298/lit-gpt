from torchmetrics import Accuracy, F1Score, Recall, Precision
from typing import Any


class ClassificationMetrics():
    def __init__(self, class_num: int) -> None:
        self.class_num = class_num
        self.accuracy = Accuracy(task="multilabel", num_classes=self.class_num)
        self.f1score = F1Score(task="multilabel", num_classes=self.class_num, average="macro")
        self.recall = Recall(task="multilabel", num_classes=self.class_num, average="macro")
        self.precision = Precision(task="multilabel", num_classes=self.class_num, average="macro")
        
        self.all = [self.accuracy, self.f1score, self.recall, self.precision]
    
    def for_all_metrics(self, function: Any, *args) -> Any:
        return [function(metric, *args) for metric in self.all]
        
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
