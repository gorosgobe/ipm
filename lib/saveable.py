import os
from abc import ABC, abstractmethod

import torch


class Saveable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_info(self):
        pass

    def save(self, path, info=None):
        if info is None:
            torch.save(self.get_info(), path)
        else:
            torch.save(info, path)


class BestSaveable(Saveable):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_best_info(self):
        pass

    def save_best_model(self, path):
        best_info = self.get_best_info()
        name = best_info["name"]
        self.save(os.path.join(path, f"{name}.pt"), best_info)
