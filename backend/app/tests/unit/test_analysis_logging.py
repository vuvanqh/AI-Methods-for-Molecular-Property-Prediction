import logging

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.analysis_logging import log_analysis
from src.db.analysis_status import AnalysisStatus


class _StubSession:
    def __init__(self, commit_error: Exception | None = None) -> None:
        self.events: list[str] = []
        self.commit_error = commit_error

    def rollback(self) -> None:
        self.events.append("rollback")

    def add(self, _obj: object) -> None:
        self.events.append("add")

    def commit(self) -> None:
        self.events.append("commit")
        if self.commit_error is not None:
            raise self.commit_error


def test_log_analysis_rolls_back_before_insert_only_for_failed_status() -> None:
    completed_db, failed_db = _StubSession(), _StubSession()
    log_analysis(completed_db, "CCO", AnalysisStatus.COMPLETED, {"top_k": 5}, {"ok": 1})  # type: ignore[arg-type]
    log_analysis(failed_db, "CCO", AnalysisStatus.FAILED, {"top_k": 5}, None)  # type: ignore[arg-type]

    assert completed_db.events == ["add", "commit"]
    assert failed_db.events == ["rollback", "add", "commit"]


@pytest.mark.parametrize("status", [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED])
def test_log_analysis_swallows_sqlalchemy_error_rolls_back_and_logs(
    status: AnalysisStatus, caplog: pytest.LogCaptureFixture
) -> None:
    db = _StubSession(commit_error=SQLAlchemyError("db down"))
    with caplog.at_level(logging.ERROR, logger="src.analysis_logging"):
        log_analysis(db, "CCO", status, {"top_k": 5}, None)  # type: ignore[arg-type]

    assert db.events[-1] == "rollback"
    assert any(
        r.levelno == logging.ERROR and "CCO" in r.getMessage() for r in caplog.records
    )


def test_log_analysis_propagates_non_sqlalchemy_errors() -> None:
    db = _StubSession(commit_error=RuntimeError("unexpected failure"))
    with pytest.raises(RuntimeError, match="unexpected failure"):
        log_analysis(db, "CCO", AnalysisStatus.COMPLETED, {"top_k": 5}, None)  # type: ignore[arg-type]
