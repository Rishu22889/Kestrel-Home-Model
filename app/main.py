"""Kestrel claim-review service.

  POST /predict          one claim as JSON -> risk score, level, reasons
  GET  /                 the review screen
  GET  /health           is the model loaded?
  GET  /options          partner ids, SKUs and example claim ids for the screen
  GET  /example/{id}     a stored claim, to prefill the screen

Starts even when the model has not been trained; /predict then returns 503
with instructions instead of crashing.
"""
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.scoring import NotReady, Scorer

app = FastAPI(title="Kestrel claim review", version="1.0")
STATIC = Path(__file__).parent / "static"

try:
    scorer, load_error = Scorer(), None
except NotReady as e:
    scorer, load_error = None, str(e)


class Claim(BaseModel):
    claim_id: Optional[str] = None
    submitted_at: Optional[str] = Field(None, description="IST, e.g. 2026-07-14 10:30; default now")
    partner_id: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    product_serial: str = Field(min_length=3)
    days_since_purchase: int = Field(ge=0)
    claim_amount_inr: float = Field(gt=0)
    photo_attached: Literal["Y", "N"]
    partner_inspected: Literal["Y", "N"]
    claim_description: str = Field(min_length=2)
    inspector_note: Optional[str] = None
    customer_prior_claims: int = Field(0, ge=0)
    source: Literal["crm", "legacy_zoho"] = "crm"

    @field_validator("submitted_at")
    @classmethod
    def _valid_time(cls, v):
        if v in (None, ""):
            return None
        try:
            pd.Timestamp(v)
        except ValueError:
            raise ValueError("must be a date-time like 2026-07-14 10:30")
        return v


def _need_model():
    if scorer is None:
        raise HTTPException(503, detail=load_error)
    return scorer


@app.get("/")
def screen():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"model_loaded": scorer is not None, "error": load_error,
            "model": scorer.meta["model"] if scorer else None}


@app.get("/options")
def options():
    s = _need_model()
    return {"partners": sorted(s.partners.index), "skus": sorted(s.products.index),
            "examples": s.example_ids()}


@app.get("/example/{claim_id}")
def example(claim_id: str):
    try:
        return _need_model().example(claim_id)
    except KeyError:
        raise HTTPException(404, detail=f"No stored claim {claim_id}")


@app.post("/predict")
def predict(claim: Claim):
    s = _need_model()
    try:
        return s.score(claim.model_dump())
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
