from fastapi import FastAPI, File, Form, UploadFile, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time

from .db import init_db
from .schemas import ManualPlanRequest, RunsResponse, UploadAndPlanResponse
from .services import list_runs, process_uploaded_file, save_manual_plan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_ai_dw_api")

app = FastAPI(title="Smart AI Data Warehouse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(
        "%s %s -> %s (%sms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "message": "Malformed request",
            "path": request.url.path,
            "errors": exc.errors(),
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-plan", response_model=UploadAndPlanResponse)
def generate_plan(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
):
    return process_uploaded_file(file, target_column)


@app.post("/upload-and-plan", response_model=UploadAndPlanResponse)
def upload_and_plan(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
):
    return process_uploaded_file(file, target_column)


@app.post("/manual-plan")
def manual_plan(payload: ManualPlanRequest):
    return save_manual_plan(
        payload.dataset_name,
        payload.target_column,
        payload.schema,
        payload.plan,
    )


@app.get("/runs", response_model=RunsResponse)
def runs(limit: int = Query(default=20, ge=1, le=100)):
    return {"runs": list_runs(limit)}