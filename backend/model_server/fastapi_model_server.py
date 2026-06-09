"""Optional DeepSeek LoRA model server for Smart AI DataWarehouse.

Run this separately from the main backend because loading the 1.3B model is heavy.

Expected folder layout:
backend/model_server/fastapi_model_server.py
deepseek_stage2/
  adapter_model.safetensors
  adapter_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json

Start:
  pip install -r backend/model_server/requirements-model.txt
  uvicorn backend.model_server.fastapi_model_server:app --host 0.0.0.0 --port 8001

Then set in backend/.env:
  MODEL_API_URL=http://localhost:8001/generate-plan
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "deepseek-ai/deepseek-coder-1.3b-base")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "deepseek_stage2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "768"))

app = FastAPI(title="Smart AI DataWarehouse DeepSeek Model API", version="1.0.0")
_tokenizer = None
_model = None


class PlanRequest(BaseModel):
    schema_description: str
    target_column: str | None = None
    dataset_name: str | None = None


def _load_model():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    adapter_path = Path(ADAPTER_PATH)
    tokenizer_source = str(adapter_path) if adapter_path.exists() else BASE_MODEL
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    _model = PeftModel.from_pretrained(base, str(adapter_path))
    _model.eval()
    return _tokenizer, _model


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    raw = match.group(0)
    # simple brace repair for truncated outputs
    diff = raw.count("{") - raw.count("}")
    if diff > 0:
        raw += "}" * diff
    return json.loads(raw)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "base_model": BASE_MODEL, "adapter_path": ADAPTER_PATH, "cuda": torch.cuda.is_available()}


@app.post("/generate-plan")
def generate_plan(req: PlanRequest) -> dict[str, Any]:
    tokenizer, model = _load_model()
    prompt = req.schema_description.strip()
    # Seed the JSON so task_type appears first, matching your training summary.
    if not prompt.endswith('{"task_type": "'):
        prompt = f'{prompt}\n{{"task_type": "'

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    completion = decoded[len(prompt):]
    reconstructed = '{"task_type": "' + completion if '{"task_type": "' not in completion[:30] else completion

    try:
        plan = _extract_json(reconstructed)
        if req.target_column and not plan.get("target_column"):
            plan["target_column"] = req.target_column
        if req.dataset_name and not plan.get("dataset_name"):
            plan["dataset_name"] = req.dataset_name
        return {"valid_json": True, "plan": plan}
    except Exception as exc:  # return raw output so backend can fall back cleanly
        return {"valid_json": False, "error": str(exc), "raw_output": completion}
