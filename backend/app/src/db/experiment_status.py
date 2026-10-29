from enum import Enum


class ExperimentStatus(str, Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"
