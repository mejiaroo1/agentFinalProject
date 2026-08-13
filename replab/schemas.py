"""Pydantic models for the paper-with-code reproduction pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    IMPOSSIBLE = "IMPOSSIBLE"
    RISKY_BUT_POSSIBLE = "RISKY_BUT_POSSIBLE"
    READY = "READY"


class PaperCandidate(BaseModel):
    title: str
    arxiv_id: str = ""
    year: int | None = None
    keywords: list[str] = Field(default_factory=list)
    abstract: str = ""
    paper_url: str = ""
    repo_url: str = ""
    stars: int | None = None
    venue: str = ""
    # Filled during search-time feasibility screening
    verdict: str = ""
    entrypoint: str = ""
    alarms: list[str] = Field(default_factory=list)
    feasibility: dict[str, Any] | None = None


class FeasibilityReport(BaseModel):
    summary: str = Field(description="Concise paper/repo summary for the user.")
    python_version: str = Field(
        default="unknown",
        description="Suggested or required Python version, if known.",
    )
    cuda_required: bool = Field(
        default=False,
        description="True if GPU/CUDA appears required with no CPU demo path.",
    )
    packages: list[str] = Field(
        default_factory=list,
        description="Key packages / dependency highlights.",
    )
    estimated_download_mb: float | None = Field(
        default=None,
        description="Rough download size in MB (repo + known datasets), if estimable.",
    )
    estimated_disk_mb: float | None = Field(
        default=None,
        description="Rough disk need in MB after install, if estimable.",
    )
    estimated_runtime_minutes: float | None = Field(
        default=None,
        description="Rough wall-clock minutes for a minimal demo run, if known.",
    )
    entrypoint: str = Field(
        default="",
        description="Suggested command to run a minimal experiment, e.g. python train.py --epochs 1",
    )
    alarms: list[str] = Field(
        default_factory=list,
        description="Warnings: large downloads, old pins, missing data, license, etc.",
    )
    verdict: Verdict = Field(description="IMPOSSIBLE | RISKY_BUT_POSSIBLE | READY")
    reasons: list[str] = Field(
        default_factory=list,
        description="Short bullet reasons supporting the verdict.",
    )
    has_requirements: bool = False
    has_readme: bool = False
    repo_size_kb: int | None = None


class RunResult(BaseModel):
    command: str = ""
    exit_code: int | None = None
    duration_sec: float | None = None
    log_tail: str = ""
    log_path: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    error: str = ""
    run_dir: str = ""
    narration: str = Field(
        default="",
        description="Plain-language success write-up (paper + entrypoint).",
    )


class ParamSpec(BaseModel):
    name: str
    default: str = ""
    description: str = ""
    cli_flag: str = ""


class TweakPlan(BaseModel):
    entrypoint: str = ""
    parameters: list[ParamSpec] = Field(default_factory=list)
    notes: str = ""
