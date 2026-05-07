"""Pydantic v2 models for CHEX - Document Intelligence."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Label(str, Enum):
    GROUNDED = "GROUNDED"
    ABSENT = "ABSENT"
    CONTRADICTS_PRIOR = "CONTRADICTS_PRIOR"


class RawCUADExample(BaseModel):
    """A single row extracted from the CUAD HuggingFace dataset."""

    contract_id: str
    contract_text: str
    question: str
    answer_text: Optional[str] = None
    answer_start: Optional[int] = None


class PerturbedExample(BaseModel):
    """Intermediate representation after applying a perturbation strategy."""

    source_contract_id: str
    perturbed_contract_text: str
    question: str
    label: Label
    original_answer_text: Optional[str] = None
    perturbed_span: Optional[str] = None


class LabeledQAExample(BaseModel):
    """The canonical training / evaluation unit."""

    contract_id: str
    contract_text: str
    question: str
    label: Label
    answer: Optional[str] = None
    citation: Optional[str] = None
    reasoning: str


class ModelOutput(BaseModel):
    """Structured JSON that the fine-tuned model must produce."""

    question: str
    label: Label
    answer: Optional[str] = None
    citation: Optional[str] = None
    reasoning: str


class BankStatementSummary(BaseModel):
    """Structured summary extracted from a bank statement."""

    total_credits: Optional[str] = None
    total_debits: Optional[str] = None
    largest_transaction: Optional[str] = None
    recurring_payments: Optional[list[str]] = None
    flags: Optional[list[str]] = None
    raw_reasoning: str


class BenchmarkResult(BaseModel):
    """Per-example result recorded during evaluation."""

    example_id: str
    predicted_label: Label
    ground_truth_label: Label
    predicted_answer: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    citation_present: bool = False
    latency_ms: float = 0.0
