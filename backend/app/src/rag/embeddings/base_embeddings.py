from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddings(ABC):
    name: str
    version: str

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self):
        raise NotImplementedError("Subclasses must implement the load method.")

    @abstractmethod
    def unload(self):
        raise NotImplementedError("Subclasses must implement the unload method.")

    @abstractmethod
    def embedd(self, smiles: str) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the embedd method.")
