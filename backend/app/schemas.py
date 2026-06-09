from typing import Any, Literal
from pydantic import BaseModel, Field


class UploadAndPlanResponse(BaseModel):
    run_id: int
    dataset_name: str
    target_column: str | None
    task_type: str | None
    schema: dict[str, Any]
    validation: dict[str, Any]
    plan: dict[str, Any]
    download_ready: bool = False


class ManualPlanRequest(BaseModel):
    dataset_name: str = Field(min_length=1)
    target_column: str | None = None
    schema: dict[str, Any]
    plan: dict[str, Any]
    execute_mode: Literal['plan_only', 'use_existing_modules'] = 'plan_only'


class RunsResponse(BaseModel):
    runs: list[dict[str, Any]]
