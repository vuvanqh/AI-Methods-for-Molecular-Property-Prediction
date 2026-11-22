import logging
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .db.analysis_status import AnalysisStatus
from .db.models import Analysis

logger = logging.getLogger(__name__)


def log_analysis(
    db: Session,
    standardized_smiles: str,
    status: AnalysisStatus,
    config: dict,
    response: dict | None = None,
) -> None:
    """Store the data that the user gets with standardized SMILES (GNN+XAI, similar compounds, LLM summary)."""
    analysis = Analysis(
        id=uuid.uuid4(),
        standardized_input_smiles=standardized_smiles,
        status=status,
        config=config,
        results=response,
    )
    try:
        if status == AnalysisStatus.FAILED:
            db.rollback()
        db.add(analysis)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Failed to log analysis for %s", standardized_smiles)
