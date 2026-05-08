"""
FastAPI server exposing CHEX analyses via HTTP.

Endpoints:
- POST /api/contract/analyse
- POST /api/bank/analyse

Both endpoints use the remote MLX chat-completions backend configured by
the `MLX_SERVER_URL` environment variable.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from data.schema import ModelOutput
from serving.mlx_inference import MLX_SERVER_URL, analyse_bank_json, analyse_contract_json

app = FastAPI(title="CHEX API", version="0.1.0")


class ContractAnalyseRequest(BaseModel):
    contract_text: str
    question: str


class BankAnalyseResponse(BaseModel):
    summaries: list[dict]
    recurring_payments_union: list[str]
    flags_union: list[str]
    answer: Optional[dict] = None
    errors: list[str] = []


@app.get("/health")
def health() -> dict:
    return {"ok": True, "mlx_server_url_set": bool(MLX_SERVER_URL)}


@app.post("/api/contract/analyse", response_model=ModelOutput)
def contract_analyse(req: ContractAnalyseRequest) -> ModelOutput:
    try:
        return analyse_contract_json(req.contract_text, req.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/bank/analyse", response_model=BankAnalyseResponse)
async def bank_analyse(
    statement_text: Annotated[Optional[str], Form()] = None,
    question: Annotated[Optional[str], Form()] = None,
    pdf_password: Annotated[Optional[str], Form()] = None,
    files: Annotated[list[UploadFile], File()] = [],
) -> BankAnalyseResponse:
    try:
        uploads: list[tuple[str, bytes]] = []
        for f in files or []:
            uploads.append((f.filename or "upload.bin", await f.read()))

        result = analyse_bank_json(
            statement_text=statement_text,
            uploads=uploads or None,
            pdf_password=pdf_password,
            question=question,
        )
        return BankAnalyseResponse(
            summaries=result["summaries"],
            recurring_payments_union=result["recurring_payments_union"],
            flags_union=result["flags_union"],
            answer=result["answer"],
            errors=result["errors"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

