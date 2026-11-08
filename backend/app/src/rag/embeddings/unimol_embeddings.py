import gc
import torch
from unimol_tools import UniMolRepr
from ..config import unimolSettings
import numpy as np
from .base_embeddings import BaseEmbeddings
import threading


class UniMolEmbeddings(BaseEmbeddings):
    name = "unimol"
    version = unimolSettings.model_name

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._lock = threading.Lock()
        self._unload_timer: threading.Timer | None = None

    def load(self):
        """
        Load the UniMol model into memory.
        """
        if self.model is None:
            self.model = UniMolRepr(
                use_cuda=torch.cuda.is_available() and self.device == "cuda",
                model_name=unimolSettings.model_name,
                remove_hs=unimolSettings.remove_hydrogens,
                batch_size=unimolSettings.batch_size,
            )
        return self.model

    def unload(self):
        """
        Unload the UniMol model from memory to free up resources.
        """
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                gc.collect()
                torch.cuda.empty_cache()

    def _restart_unload_timer(self):
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(
            unimolSettings.idle_unload_seconds, self.unload
        )
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def embedd(self, smiles: str) -> np.ndarray:
        """
        Generate embeddings for a given SMILES string using the UniMol model.

        Args:
            smiles (str): The SMILES string of the molecule.
        """

        with self._lock:
            model = self.load()
            embeddings = np.array(model.get_repr([smiles]), dtype=np.float32)[0]
            self._restart_unload_timer()

        return embeddings


unimol_embeddings = UniMolEmbeddings()
