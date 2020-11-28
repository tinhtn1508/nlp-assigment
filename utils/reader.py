from itertools import islice
from typing import (
    List,
)
import os
import subprocess

class BatchReader():
    def __init__(self, path_file_name: int, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.path_file_name = path_file_name
        self.batch_size = batch_size
        self.root_file = open(path_file_name, 'r')

    def __del__(self) -> None:
        self.root_file.close()

    def getDataBatch(self, batch: int) -> List[str]:
        assert self.root_file != None
        assert self.num_batches != batch
        return list(islice(self.root_file, 0, self.batch_size))

    def resetBatch(self) -> None:
        assert self.root_file != None
        self.root_file.close()
        self.root_file = open(self.path_file_name, 'r')

def getNumberOfLine(path_file_name: str) -> int:
    output = subprocess.check_output("wc -l " + path_file_name + " | awk \'{print $1}\'", shell=True, universal_newlines=True)
    return int(output)
