from enum import Enum


class RetrievalType(str, Enum):
    FINGERPRINT = "fingerprint"
    SEMANTIC = "semantic"
    FUSION = "fusion"
