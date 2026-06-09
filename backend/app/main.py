import logging
import os
import time
import requests

from fastapi import FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from . import supabase_client as sb
from .db import init_db
from .schemas import DatasetsResponse, ManualPlanRequest, RunsResponse, UploadAndPlanResponse
from .services import list_datasets, list_runs, process_uploaded_file, save_manual_plan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_ai_dw_api")

app = FastAPI(title="Smart AI Data Warehouse API", version="2.0.0")
origins = os.getenv("APP_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup() -> None:
    init_db()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:  # keep frontend readable instead of a giant stack trace
        logger.exception("Unhandled backend error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"message": "Backend error", "detail": str(exc), "path": request.url.path})
    logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, round((time.time() - start_time) * 1000, 2))
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"message": "Malformed request", "path": request.url.path, "errors": exc.errors()})

@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Smart AI DataWarehouse API is running. Open /docs for API docs."}

@app.get("/health")
def health() -> dict:
    model_url = os.getenv("MODEL_API_URL", "").strip()
    model = {"enabled": bool(model_url), "url": model_url or None, "reachable": False}
    if model_url:
        try:
            root_url = model_url.rsplit("/", 1)[0]
            r = requests.get(root_url + "/", timeout=5)
            model["reachable"] = r.ok
        except Exception as exc:
            model["reason"] = str(exc)
    return {"status": "ok", "supabase": sb.status(), "model_api": model}

def _owner(authorization: str | None) -> str | None:
    return sb.decode_user_id(authorization)

@app.post("/upload-and-plan", response_model=UploadAndPlanResponse)
def upload_and_plan(file: UploadFile = File(...), target_column: str | None = Form(default=None), authorization: str | None = Header(default=None)):
    return process_uploaded_file(file, target_column, _owner(authorization))

@app.post("/generate-plan", response_model=UploadAndPlanResponse)
def generate_plan(file: UploadFile = File(...), target_column: str | None = Form(default=None), authorization: str | None = Header(default=None)):
    return process_uploaded_file(file, target_column, _owner(authorization))

@app.post("/manual-plan")
def manual_plan(payload: ManualPlanRequest, authorization: str | None = Header(default=None)):
    return save_manual_plan(payload.dataset_name, payload.target_column, payload.schema, payload.plan, _owner(authorization))

@app.get("/download-curated/{dataset_name}")
def download_curated(dataset_name: str):
    import os
    from pathlib import Path
    # Anchor to project root (two levels up from this file)
    project_root = Path(__file__).resolve().parents[2]
    curated_path = project_root / "data" / "curated" / f"{dataset_name}_curated.csv"
    if not curated_path.exists():
        return JSONResponse(status_code=404, content={"message": f"Curated file not found for dataset: {dataset_name}"})
    return FileResponse(
        path=str(curated_path),
        media_type="text/csv",
        filename=f"{dataset_name}_curated.csv",
    )

@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = Query(default=20, ge=1, le=100), authorization: str | None = Header(default=None)):
    return {"runs": list_runs(limit, _owner(authorization))}

@app.get("/datasets", response_model=DatasetsResponse)
def datasets(limit: int = Query(default=50, ge=1, le=100), authorization: str | None = Header(default=None)):
    return {"datasets": list_datasets(limit, _owner(authorization))}
