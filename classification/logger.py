from pathlib import Path
from typing import List

class CSVLogger():
    def __init__(self, log_dir: Path, log_file: str = "logs.csv") -> None:
        self.log_dir = log_dir
        self.log_filepath = self.log_dir / log_file
        
        self.log_filepath.parents
        
    def log(self, data: List):
        pass