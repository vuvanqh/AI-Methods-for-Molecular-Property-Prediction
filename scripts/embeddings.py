from functools import cache

import numpy as np
from unimol_tools import UniMolRepr

EMBEDDING_MODEL = "unimol"


@cache
def _model(version: str) -> UniMolRepr:
    # weights download from Hugging Face (dptech/Uni-Mol-Models) on first use, then come from the local cache
    return UniMolRepr(data_type="molecule", model_name=version, remove_hs=False)


def unimol_embeddings(smiles: list[str], version: str = "unimolv1") -> np.ndarray:
    """
    Shape (len(smiles), 512) for unimolv1.
    """
    return np.asarray(_model(version).get_repr(smiles), dtype=np.float32)


if __name__ == "__main__":
    vectors = unimol_embeddings(["CCO", "c1ccccc1"])
    assert vectors.shape == (2, 512), vectors.shape
    assert not np.allclose(vectors[0], vectors[1])
    print("embeddings self-check passed")
