"""Pydantic models for permission/policy verdicts."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    COVERED = "covered"
    UNCLEAR = "unclear"
    MISMATCH = "mismatch"


class PermissionVerdict(BaseModel):
    permission: str = Field(..., description="Android permission identifier")
    data_access: str = Field(..., description="What data this permission grants access to")
    policy_mention: str = Field(..., description="Quoted span from policy or 'NOT MENTIONED'")
    verdict: Verdict
    reasoning: str = Field(..., description="One-sentence justification")


class AppReport(BaseModel):
    app_id: str
    app_title: str
    model: str
    permissions: List[PermissionVerdict]
    summary: str
    policy_chars: int = 0
    policy_url: Optional[str] = None
    error: Optional[str] = None

    @property
    def covered_count(self) -> int:
        return sum(1 for p in self.permissions if p.verdict == Verdict.COVERED)

    @property
    def unclear_count(self) -> int:
        return sum(1 for p in self.permissions if p.verdict == Verdict.UNCLEAR)

    @property
    def mismatch_count(self) -> int:
        return sum(1 for p in self.permissions if p.verdict == Verdict.MISMATCH)
