"""
Plan Dispatcher - Transformer-first preprocessing plan controller

Purpose:
    Runs after schema extraction and before validation/transformation/feature engineering.

Final design:
    1. Read schema JSON files from data/schema/
    2. Build a schema prompt without _meta
    3. Try DeepSeek base model + LoRA adapter from models/deepseek_stage2/
    4. Parse, repair, normalize, and validate transformer output
    5. If transformer output is invalid/incomplete/logically weak:
         use the existing rule-based PlanGenerator as fallback
    6. If PlanGenerator cannot run because of Supabase/import issues:
         use internal emergency fallback
    7. Always save one clean normalized plan per dataset in metadata/plans/

Important:
    - _meta is never treated as a real dataset column.
    - The final saved plan has one normalized contract.
    - Hallucinated blocks from the model such as ingestion/schema/validation/transformation/export_paths are ignored.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Optional model imports
# ============================================================

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    MODEL_LIBS_AVAILABLE = True
except Exception:
    MODEL_LIBS_AVAILABLE = False


# ============================================================
# Paths / constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_DIR = PROJECT_ROOT / "data" / "schema"
PLAN_DIR = PROJECT_ROOT / "metadata" / "plans"
DEBUG_DIR = PROJECT_ROOT / "metadata" / "debug" / "model_outputs"

BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
LORA_DIR = PROJECT_ROOT / "models" / "deepseek_stage2"

NUMERIC_DTYPES = {
    "int64",
    "float64",
    "int32",
    "float32",
    "int",
    "float",
    "double",
}

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 2048
DO_SAMPLE = False
REPETITION_PENALTY = 1.1

_model = None
_tokenizer = None


# ============================================================
# JSON helpers
# ============================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_raw_model_output(dataset_name: str, raw_output: Optional[str]) -> Optional[str]:
    if not raw_output:
        return None

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")
    output_path = DEBUG_DIR / f"{safe_name}_raw_model_output.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_output)

    return str(output_path)


# ============================================================
# Schema helpers
# ============================================================

def real_schema_items(schema: dict) -> List[Tuple[str, dict]]:
    """
    Return real dataset columns only.
    _meta is metadata, not a dataframe column.
    """
    return [
        (col_name, col_info)
        for col_name, col_info in schema.items()
        if col_name != "_meta" and isinstance(col_info, dict)
    ]


def existing_columns(schema: dict) -> List[str]:
    return [col_name for col_name, _ in real_schema_items(schema)]


def is_numeric_dtype(dtype: str) -> bool:
    dtype = str(dtype).lower()
    return dtype in NUMERIC_DTYPES or "int" in dtype or "float" in dtype or "double" in dtype


def looks_like_id(col_name: str) -> bool:
    cleaned = col_name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return (
        cleaned == "id"
        or cleaned.endswith("id")
        or cleaned.startswith("unnamed")
        or "passengerid" in cleaned
        or "customerid" in cleaned
        or "employeeid" in cleaned
        or "employeenumber" in cleaned
    )


def looks_like_date(col_name: str, dtype: str = "") -> bool:
    col = col_name.strip().lower()
    dtype = str(dtype).lower()

    return (
        "date" in col
        or "time" in col
        or "timestamp" in col
        or "datetime" in dtype
        or "created_at" in col
        or "updated_at" in col
    )


def is_numeric_column(schema: dict, col_name: str) -> bool:
    info = schema.get(col_name, {})
    dtype = str(info.get("dtype", "")).lower()
    return is_numeric_dtype(dtype)


def is_categorical_column(schema: dict, col_name: str) -> bool:
    return col_name in existing_columns(schema) and not is_numeric_column(schema, col_name)


def filter_existing_columns(
    columns: Any,
    schema: dict,
    target_column: Optional[str] = None,
    require_type: Optional[str] = None,
    excluded: Optional[List[str]] = None,
) -> List[str]:
    """
    Keep only columns that exist in the schema.
    Never include target column.
    Optionally enforce numeric/categorical type.
    """
    if not isinstance(columns, list):
        return []

    valid = set(existing_columns(schema))
    excluded = set(excluded or [])
    filtered = []

    for col in columns:
        if not isinstance(col, str):
            continue

        if col not in valid:
            continue

        if target_column and col == target_column:
            continue

        if col in excluded:
            continue

        if require_type == "numeric" and not is_numeric_column(schema, col):
            continue

        if require_type == "categorical" and not is_categorical_column(schema, col):
            continue

        if col not in filtered:
            filtered.append(col)

    return filtered


# ============================================================
# Target / task helpers
# ============================================================

def normalize_col_name(col_name: str) -> str:
    return str(col_name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def target_candidate_score(schema: dict, col_name: str) -> int:
    """
    Score possible targets. Higher is better.
    This is used for validation and emergency fallback.
    It does not blindly trust schema['_meta']['target_column'].
    """
    if col_name not in schema or col_name == "_meta":
        return -9999

    info = schema.get(col_name, {})
    dtype = str(info.get("dtype", "")).lower()
    unique_count = int(info.get("unique_count", 0) or 0)

    lowered = col_name.strip().lower()
    compact = normalize_col_name(col_name)

    if looks_like_id(col_name) or looks_like_date(col_name, dtype):
        return -500

    bad_words = [
        "filename",
        "file",
        "path",
        "url",
        "description",
        "comment",
        "notes",
        "competitors",
    ]

    for word in bad_words:
        if word in lowered:
            return -120

    score = 0

    strong_exact = {
        "target",
        "label",
        "class",
        "species",
        "price",
        "sales",
        "churn",
        "attrition",
        "cardio",
        "survived",
        "mpg",
        "expenses",
        "charges",
        "area",
        "diagnosis",
        "score",
    }

    if compact in {normalize_col_name(x) for x in strong_exact}:
        score += 120

    strong_contains = [
        "target",
        "label",
        "class",
        "species",
        "price",
        "sales",
        "churn",
        "attrition",
        "cardio",
        "survived",
        "mpg",
        "expenses",
        "charges",
        "diagnosis",
        "admit",
        "admission",
        "outcome",
        "result",
    ]

    for word in strong_contains:
        if word in lowered:
            score += 80
            break

    if "chance" in lowered and "admit" in lowered:
        score += 140

    columns = existing_columns(schema)
    if columns and col_name == columns[-1]:
        score += 35

    if not is_numeric_dtype(dtype):
        score += 15

    if 2 <= unique_count <= 10:
        score += 20

    if is_numeric_dtype(dtype) and unique_count > 10:
        score += 10

    if unique_count > 1000 and not is_numeric_dtype(dtype):
        score -= 80

    return score


def best_target_candidate(schema: dict) -> Optional[str]:
    cols = existing_columns(schema)
    if not cols:
        return None

    scored = [(col, target_candidate_score(schema, col)) for col in cols]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_col, best_score = scored[0]

    if best_score > -100:
        return best_col

    non_id_non_date = []
    for col, info in real_schema_items(schema):
        dtype = str(info.get("dtype", "")).lower()
        if looks_like_id(col) or looks_like_date(col, dtype):
            continue
        non_id_non_date.append(col)

    if non_id_non_date:
        return non_id_non_date[-1]

    return cols[-1]


def infer_task_type(schema: dict, target_column: Optional[str]) -> str:
    if not target_column or target_column not in schema:
        return "classification"

    target_info = schema.get(target_column, {})
    dtype = str(target_info.get("dtype", "")).lower()
    unique_count = int(target_info.get("unique_count", 0) or 0)

    if is_numeric_dtype(dtype) and unique_count > 10:
        return "regression"

    return "classification"


def choose_target(model_target: Optional[str], schema: dict) -> str:
    if model_target and model_target in existing_columns(schema):
        return model_target

    return best_target_candidate(schema) or existing_columns(schema)[-1]


def choose_task_type(raw_task_type: Optional[str], schema: dict, target_column: str) -> str:
    """
    Final task type is corrected from the target dtype/cardinality.
    This prevents cases like sales/expenses/Chance of Admit being classification.
    """
    inferred = infer_task_type(schema, target_column)

    if raw_task_type not in {"classification", "regression"}:
        return inferred

    if raw_task_type != inferred:
        return inferred

    return raw_task_type


# ============================================================
# Prompt builder
# ============================================================

def build_schema_description(schema: dict, dataset_name: str) -> str:
    """
    Build the prompt expected by the transformer.

    Rules:
        - Include all real columns.
        - Do not include _meta.
        - Do not preselect/remove a target.
        - Numeric columns include mean/min/max.
        - Categorical columns include up to 3 samples.
    """
    meta = schema.get("_meta", {}) if isinstance(schema.get("_meta", {}), dict) else {}

    rows = meta.get("row_count", "?")
    cols = len(real_schema_items(schema))
    column_names = [col for col, _ in real_schema_items(schema)]

    soft_target = best_target_candidate(schema)
    task_hint = infer_task_type(schema, soft_target)

    lines = [
        f"Generate preprocessing steps for a {task_hint} task.",
        "",
        f"Dataset: {dataset_name}",
        f"Shape: {rows} rows x {cols} columns",
        "",
        f"All columns: {', '.join(column_names)}",
        "",
        "Column details:",
    ]

    for col_name, col_info in real_schema_items(schema):
        dtype = str(col_info.get("dtype", "unknown"))
        unique_count = col_info.get("unique_count", "?")
        missing_percentage = col_info.get("missing_percentage", 0)

        if is_numeric_dtype(dtype):
            mean = col_info.get("mean", "unknown")
            min_val = col_info.get("min", "unknown")
            max_val = col_info.get("max", "unknown")

            lines.append(
                f"  - {col_name}: {dtype}, {unique_count} unique, "
                f"{missing_percentage}% missing, mean={mean}, min={min_val}, max={max_val}"
            )
        else:
            samples = col_info.get("sample_values", [])
            if isinstance(samples, list):
                samples = samples[:3]
            else:
                samples = []

            lines.append(
                f"  - {col_name}: {dtype}, {unique_count} unique, "
                f"{missing_percentage}% missing, samples={samples}"
            )

    return "\n".join(lines)


# ============================================================
# Model loading / inference
# ============================================================

def model_available() -> bool:
    if not MODEL_LIBS_AVAILABLE:
        return False

    if not LORA_DIR.exists():
        return False

    return True


def load_model():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not MODEL_LIBS_AVAILABLE:
        raise RuntimeError(
            "Model libraries are not installed. Install transformers, peft, accelerate, sentencepiece, and torch."
        )

    if not LORA_DIR.exists():
        raise FileNotFoundError(f"LoRA weights not found at: {LORA_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"   Loading tokenizer from: {LORA_DIR}")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            str(LORA_DIR),
            trust_remote_code=True,
        )
    except Exception:
        print("   Could not load tokenizer from LoRA folder. Trying base model tokenizer.")
        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _tokenizer.padding_side = "right"

    print(f"   Loading base model: {BASE_MODEL}")
    print(f"   Device: {device}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    print(f"   Applying LoRA adapter: {LORA_DIR}")
    _model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    _model.eval()

    print("   Model loaded successfully")

    return _model, _tokenizer


def run_model_inference(schema_description: str) -> Optional[str]:
    if not model_available():
        print("   Model not available locally. Transformer step will use fallback.")
        return None

    try:
        model, tokenizer = load_model()
        device = next(model.parameters()).device

        prompt = (
            "### Instruction:\n"
            f"{schema_description}\n\n"
            "Return only valid JSON with task_type, target_column, preprocessing, split_strategy, and smote.\n"
            "Do not include ingestion, schema, validation, transformation, or export_paths.\n\n"
            "### Response:\n"
            '{"task_type": "'
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return '{"task_type": "' + generated

    except Exception as e:
        print(f"   Model inference failed: {e}")
        return None


# ============================================================
# JSON parsing / repair
# ============================================================

def close_unbalanced_json(raw: str) -> str:
    text = raw.strip()

    open_braces = text.count("{")
    close_braces = text.count("}")

    open_brackets = text.count("[")
    close_brackets = text.count("]")

    if close_brackets < open_brackets:
        text += "]" * (open_brackets - close_brackets)

    if close_braces < open_braces:
        text += "}" * (open_braces - close_braces)

    return text


def extract_first_json_block(raw_text: str) -> Optional[str]:
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(raw_text)):
        ch = raw_text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

        if depth == 0:
            return raw_text[start:i + 1]

    return None


def extract_preprocessing_plan_block(raw_text: str) -> dict:
    match = re.search(r'"preprocessing_plan"\s*:\s*(\{.*)', raw_text, re.DOTALL)
    if not match:
        return {}

    fragment = match.group(1)

    for end in range(len(fragment), 0, -1):
        chunk = fragment[:end].rstrip().rstrip(",")

        repaired = chunk
        repaired += "]" * max(chunk.count("[") - chunk.count("]"), 0)
        repaired += "}" * max(chunk.count("{") - chunk.count("}"), 0)

        try:
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue

    return {}


def parse_model_output(raw_output: Optional[str]) -> Tuple[dict, str]:
    if not raw_output:
        return {}, "fallback_no_model_output"

    raw = raw_output.strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}, "json_direct"
    except Exception:
        pass

    try:
        repaired = close_unbalanced_json(raw)
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else {}, "json_bracket_repair"
    except Exception:
        pass

    try:
        block = extract_preprocessing_plan_block(raw)
        if block:
            return {"preprocessing_plan": block}, "json_preprocessing_plan_block"
    except Exception:
        pass

    try:
        block_text = extract_first_json_block(raw)
        if block_text:
            parsed = json.loads(block_text)
            return parsed if isinstance(parsed, dict) else {}, "json_regex_block"
    except Exception:
        pass

    fallback = {}

    target_match = re.search(r'"target_column"\s*:\s*"([^"]+)"', raw)
    task_match = re.search(r'"task_type"\s*:\s*"([^"]+)"', raw)

    if target_match:
        fallback["target_column"] = target_match.group(1)

    if task_match:
        fallback["task_type"] = task_match.group(1)

    if fallback:
        return fallback, "regex_critical_fields"

    return {}, "fallback_parse_failed"


# ============================================================
# Plan normalization
# ============================================================

def list_from_config(config: dict, key: str) -> Optional[list]:
    if not isinstance(config, dict):
        return None

    value = config.get(key)
    return value if isinstance(value, list) else None


def normalize_transformer_plan(
    raw_plan: dict,
    schema: dict,
    dataset_name: str,
    parse_status: str,
    raw_output_path: Optional[str],
) -> dict:
    raw_plan = raw_plan if isinstance(raw_plan, dict) else {}

    preprocessing_plan = raw_plan.get("preprocessing_plan", {})
    if not isinstance(preprocessing_plan, dict):
        preprocessing_plan = {}

    model_target = (
        raw_plan.get("target_column")
        or preprocessing_plan.get("target_column")
    )

    target_column = choose_target(model_target, schema)

    raw_task_type = (
        raw_plan.get("task_type")
        or preprocessing_plan.get("task_type")
    )

    task_type = choose_task_type(raw_task_type, schema, target_column)

    preprocessing = preprocessing_plan.get("preprocessing")
    if not isinstance(preprocessing, dict):
        preprocessing = raw_plan.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        preprocessing = {}

    normalized = build_normalized_plan_from_preprocessing(
        schema=schema,
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type,
        preprocessing=preprocessing,
        split_strategy=preprocessing_plan.get("split_strategy", raw_plan.get("split_strategy", {})),
        smote=preprocessing_plan.get("smote", raw_plan.get("smote", {})),
        source="transformer",
        parse_status=parse_status,
        extra_debug={
            "raw_model_output_path": raw_output_path,
            "ignored_model_blocks": [
                "ingestion",
                "schema",
                "validation",
                "transformation",
                "export_paths",
            ],
        },
    )

    return normalized


def normalize_plan_generator_output(
    legacy_plan: dict,
    schema: dict,
    dataset_name: str,
    fallback_reason: str,
) -> dict:
    legacy_plan = legacy_plan if isinstance(legacy_plan, dict) else {}

    target_column = choose_target(legacy_plan.get("target_column"), schema)
    task_type = choose_task_type(legacy_plan.get("task_type"), schema, target_column)

    legacy_preprocessing = legacy_plan.get("preprocessing", {})
    if not isinstance(legacy_preprocessing, dict):
        legacy_preprocessing = {}

    preprocessing = {
        "drop_columns": legacy_preprocessing.get("drop_columns", []),
        "imputation": {
            "numeric_strategy": "median",
            "numeric_columns": legacy_preprocessing.get("impute_numeric", []),
            "categorical_strategy": "most_frequent",
            "categorical_columns": legacy_preprocessing.get("impute_categorical", []),
        },
        "encoding": {
            "one_hot_columns": legacy_preprocessing.get("one_hot_encode", []),
            "label_encoding_columns": legacy_preprocessing.get("label_encode", []),
        },
        "scaling": {
            "strategy": "standard",
            "columns": legacy_preprocessing.get("scale_columns", []),
        },
        "log_transform": legacy_preprocessing.get("log_transform", []),
    }

    normalized = build_normalized_plan_from_preprocessing(
        schema=schema,
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type,
        preprocessing=preprocessing,
        split_strategy=legacy_plan.get("split_strategy", {}),
        smote=legacy_plan.get("smote", {}),
        source="plan_generator_fallback",
        parse_status="plan_generator_output",
        extra_debug={
            "fallback_reason": fallback_reason,
            "legacy_plan_generator_used": True,
        },
    )

    return normalized


def build_internal_safe_fallback(
    schema: dict,
    dataset_name: str,
    fallback_reason: str,
) -> dict:
    target_column = best_target_candidate(schema) or existing_columns(schema)[-1]
    task_type = infer_task_type(schema, target_column)

    normalized = build_normalized_plan_from_preprocessing(
        schema=schema,
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type,
        preprocessing={},
        split_strategy={},
        smote={},
        source="internal_safe_fallback",
        parse_status="internal_defaults",
        extra_debug={
            "fallback_reason": fallback_reason,
            "legacy_plan_generator_used": False,
        },
    )

    return normalized


def build_normalized_plan_from_preprocessing(
    schema: dict,
    dataset_name: str,
    target_column: str,
    task_type: str,
    preprocessing: dict,
    split_strategy: Any,
    smote: Any,
    source: str,
    parse_status: str,
    extra_debug: Optional[dict] = None,
) -> dict:
    preprocessing = preprocessing if isinstance(preprocessing, dict) else {}

    all_numeric_cols = []
    all_categorical_cols = []
    missing_numeric_cols = []
    missing_categorical_cols = []
    fallback_drop_cols = []

    for col_name, col_info in real_schema_items(schema):
        if col_name == target_column:
            continue

        dtype = str(col_info.get("dtype", "")).lower()
        missing_percentage = float(col_info.get("missing_percentage", 0) or 0)

        if looks_like_id(col_name) or looks_like_date(col_name, dtype):
            fallback_drop_cols.append(col_name)
            continue

        if is_numeric_dtype(dtype):
            all_numeric_cols.append(col_name)
            if missing_percentage > 0:
                missing_numeric_cols.append(col_name)
        else:
            all_categorical_cols.append(col_name)
            if missing_percentage > 0:
                missing_categorical_cols.append(col_name)

    # Drop columns
    raw_drop_columns = preprocessing.get("drop_columns")
    if isinstance(raw_drop_columns, list):
        drop_columns = filter_existing_columns(raw_drop_columns, schema, target_column)
    else:
        drop_columns = filter_existing_columns(fallback_drop_cols, schema, target_column)

    # Imputation
    imputation = preprocessing.get("imputation", {})
    if not isinstance(imputation, dict):
        imputation = {}

    numeric_strategy = str(imputation.get("numeric_strategy", "median")).lower()
    if numeric_strategy not in {"median", "mean"}:
        numeric_strategy = "median"

    categorical_strategy = str(imputation.get("categorical_strategy", "most_frequent")).lower()
    if categorical_strategy == "mode":
        categorical_strategy = "most_frequent"
    if categorical_strategy not in {"most_frequent", "constant"}:
        categorical_strategy = "most_frequent"

    raw_numeric_impute_cols = list_from_config(imputation, "numeric_columns")
    raw_categorical_impute_cols = list_from_config(imputation, "categorical_columns")

    numeric_impute_cols = (
        filter_existing_columns(raw_numeric_impute_cols, schema, target_column, require_type="numeric", excluded=drop_columns)
        if raw_numeric_impute_cols is not None
        else [c for c in missing_numeric_cols if c not in drop_columns]
    )

    categorical_impute_cols = (
        filter_existing_columns(raw_categorical_impute_cols, schema, target_column, require_type="categorical", excluded=drop_columns)
        if raw_categorical_impute_cols is not None
        else [c for c in missing_categorical_cols if c not in drop_columns]
    )

    # Encoding
    encoding = preprocessing.get("encoding", {})
    if not isinstance(encoding, dict):
        encoding = {}

    if "one_hot_columns" in encoding or "label_encoding_columns" in encoding:
        one_hot_columns = filter_existing_columns(
            encoding.get("one_hot_columns", []),
            schema,
            target_column,
            require_type="categorical",
            excluded=drop_columns,
        )

        label_encoding_columns = filter_existing_columns(
            encoding.get("label_encoding_columns", []),
            schema,
            target_column,
            require_type="categorical",
            excluded=drop_columns,
        )
    else:
        one_hot_columns = []
        label_encoding_columns = [c for c in all_categorical_cols if c not in drop_columns]

    label_encoding_columns = [
        col for col in label_encoding_columns
        if col not in one_hot_columns
    ]

    # Scaling
    scaling = preprocessing.get("scaling", {})
    if not isinstance(scaling, dict):
        scaling = {}

    scaling_strategy = str(scaling.get("strategy", "standard")).lower()
    if scaling_strategy not in {"standard", "minmax", "none"}:
        scaling_strategy = "standard"

    raw_scaling_cols = list_from_config(scaling, "columns")

    scaling_columns = (
        filter_existing_columns(raw_scaling_cols, schema, target_column, require_type="numeric", excluded=drop_columns)
        if raw_scaling_cols is not None
        else [c for c in all_numeric_cols if c not in drop_columns]
    )

    if scaling_strategy == "none":
        scaling_columns = []

    log_transform_columns = filter_existing_columns(
        preprocessing.get("log_transform", []),
        schema,
        target_column,
        require_type="numeric",
        excluded=drop_columns,
    )

    # Split strategy
    if not isinstance(split_strategy, dict):
        split_strategy = {}

    split_strategy = {
        "test_size": split_strategy.get("test_size", 0.2),
        "random_state": split_strategy.get("random_state", 42),
        "stratify": split_strategy.get("stratify", task_type == "classification"),
    }

    # SMOTE
    if not isinstance(smote, dict):
        smote = {}

    smote = {
        "enabled": bool(smote.get("enabled", False)),
        "reason": smote.get(
            "reason",
            "Safe default: disabled unless class imbalance is explicitly confirmed"
        ),
    }

    debug = {
        "source": source,
        "parse_status": parse_status,
        "target_score": target_candidate_score(schema, target_column),
        "task_type_inferred_from_schema": infer_task_type(schema, target_column),
    }

    if extra_debug:
        debug.update(extra_debug)

    return {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_column,
        "preprocessing": {
            "drop_columns": drop_columns,
            "imputation": {
                "numeric_strategy": numeric_strategy,
                "numeric_columns": numeric_impute_cols,
                "categorical_strategy": categorical_strategy,
                "categorical_columns": categorical_impute_cols,
                "constant_value": imputation.get("constant_value", "Unknown"),
            },
            "encoding": {
                "one_hot_columns": one_hot_columns,
                "label_encoding_columns": label_encoding_columns,
            },
            "scaling": {
                "strategy": scaling_strategy,
                "columns": scaling_columns,
            },
            "log_transform": log_transform_columns,
        },
        "split_strategy": split_strategy,
        "smote": smote,
        "_debug": debug,
    }


# ============================================================
# Plan validation / fallback
# ============================================================

def validate_final_plan(plan: dict, schema: dict, strict_transformer: bool = False) -> Tuple[bool, List[str]]:
    reasons = []

    if not isinstance(plan, dict):
        return False, ["Plan is not a dictionary"]

    target = plan.get("target_column")
    if not target:
        reasons.append("Missing target_column")
    elif target not in existing_columns(schema):
        reasons.append(f"Target column does not exist in schema: {target}")
    elif looks_like_id(target) or looks_like_date(target, str(schema.get(target, {}).get("dtype", ""))):
        reasons.append(f"Target looks like ID/date column: {target}")

    task_type = plan.get("task_type")
    if task_type not in {"classification", "regression"}:
        reasons.append(f"Invalid task_type: {task_type}")

    if target and target in existing_columns(schema):
        expected_task = infer_task_type(schema, target)
        if task_type != expected_task:
            reasons.append(f"Task type mismatch: got {task_type}, expected {expected_task}")

        chosen_score = target_candidate_score(schema, target)
        best_target = best_target_candidate(schema)
        best_score = target_candidate_score(schema, best_target) if best_target else chosen_score

        if best_target and best_score >= 80 and chosen_score < best_score - 60:
            reasons.append(
                f"Target '{target}' is much weaker than likely target '{best_target}'"
            )

    preprocessing = plan.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        reasons.append("preprocessing block is missing or invalid")
        return False, reasons

    protected_lists = [
        preprocessing.get("drop_columns", []),
        preprocessing.get("imputation", {}).get("numeric_columns", []),
        preprocessing.get("imputation", {}).get("categorical_columns", []),
        preprocessing.get("encoding", {}).get("one_hot_columns", []),
        preprocessing.get("encoding", {}).get("label_encoding_columns", []),
        preprocessing.get("scaling", {}).get("columns", []),
    ]

    for columns in protected_lists:
        if isinstance(columns, list) and target in columns:
            reasons.append(f"Target column appears inside preprocessing list: {target}")

    parse_status = plan.get("_debug", {}).get("parse_status")
    if strict_transformer and parse_status in {
        "fallback_no_model_output",
        "fallback_parse_failed",
        "regex_critical_fields",
    }:
        reasons.append(f"Transformer output incomplete or weak: {parse_status}")

    return len(reasons) == 0, reasons


def generate_plan_generator_fallback(schema: dict, dataset_name: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try the existing rule-based PlanGenerator as fallback.
    Import is lazy because plan_generator imports Supabase at module load time.
    """
    try:
        from pipelines.plan_generator import PlanGenerator

        generator = PlanGenerator()
        legacy_plan = generator.generate_plan(schema=schema, dataset_name=dataset_name)

        return legacy_plan, None

    except Exception as e:
        return None, str(e)


# ============================================================
# Dispatcher
# ============================================================

def dispatch_schema(schema_file: Path) -> dict:
    dataset_name = schema_file.stem.replace("_schema", "")
    schema = load_json(schema_file)

    schema_description = build_schema_description(schema, dataset_name)

    raw_output = run_model_inference(schema_description)
    raw_output_path = save_raw_model_output(dataset_name, raw_output)

    raw_plan, parse_status = parse_model_output(raw_output)

    transformer_plan = normalize_transformer_plan(
        raw_plan=raw_plan,
        schema=schema,
        dataset_name=dataset_name,
        parse_status=parse_status,
        raw_output_path=raw_output_path,
    )

    transformer_valid, transformer_reasons = validate_final_plan(
        transformer_plan,
        schema,
        strict_transformer=True,
    )

    if transformer_valid:
        final_plan = transformer_plan
        final_plan["_debug"]["final_decision"] = "accepted_transformer_plan"
    else:
        fallback_reason = "; ".join(transformer_reasons)
        print(f"   Transformer plan rejected: {fallback_reason}")
        print("   Trying PlanGenerator fallback.")

        legacy_plan, legacy_error = generate_plan_generator_fallback(schema, dataset_name)

        if legacy_plan is not None:
            fallback_plan = normalize_plan_generator_output(
                legacy_plan=legacy_plan,
                schema=schema,
                dataset_name=dataset_name,
                fallback_reason=fallback_reason,
            )

            fallback_valid, fallback_reasons = validate_final_plan(
                fallback_plan,
                schema,
                strict_transformer=False,
            )

            if fallback_valid:
                final_plan = fallback_plan
                final_plan["_debug"]["final_decision"] = "used_plan_generator_fallback"
                final_plan["_debug"]["transformer_rejection_reasons"] = transformer_reasons
            else:
                emergency_reason = (
                    f"PlanGenerator fallback invalid: {'; '.join(fallback_reasons)}"
                )
                print(f"   {emergency_reason}")
                final_plan = build_internal_safe_fallback(
                    schema=schema,
                    dataset_name=dataset_name,
                    fallback_reason=emergency_reason,
                )
                final_plan["_debug"]["final_decision"] = "used_internal_safe_fallback"
                final_plan["_debug"]["transformer_rejection_reasons"] = transformer_reasons
                final_plan["_debug"]["plan_generator_rejection_reasons"] = fallback_reasons
        else:
            emergency_reason = f"PlanGenerator fallback failed: {legacy_error}"
            print(f"   {emergency_reason}")
            final_plan = build_internal_safe_fallback(
                schema=schema,
                dataset_name=dataset_name,
                fallback_reason=emergency_reason,
            )
            final_plan["_debug"]["final_decision"] = "used_internal_safe_fallback"
            final_plan["_debug"]["transformer_rejection_reasons"] = transformer_reasons
            final_plan["_debug"]["plan_generator_error"] = legacy_error

    output_path = PLAN_DIR / f"{dataset_name}_plan.json"
    save_json(final_plan, output_path)

    print(f"   Saved plan: {output_path.name}")
    print(f"      Final source: {final_plan['_debug']['source']}")
    print(f"      Decision:     {final_plan['_debug'].get('final_decision')}")
    print(f"      Parse:        {final_plan['_debug']['parse_status']}")
    print(f"      Target:       {final_plan['target_column']}")
    print(f"      Task:         {final_plan['task_type']}")
    print(f"      Drop:         {final_plan['preprocessing']['drop_columns']}")
    print(f"      Numeric impute: {final_plan['preprocessing']['imputation']['numeric_strategy']} -> "
          f"{final_plan['preprocessing']['imputation']['numeric_columns']}")
    print(f"      Categorical impute: {final_plan['preprocessing']['imputation']['categorical_strategy']} -> "
          f"{final_plan['preprocessing']['imputation']['categorical_columns']}")
    print(f"      Encode one-hot: {final_plan['preprocessing']['encoding']['one_hot_columns']}")
    print(f"      Encode label:   {final_plan['preprocessing']['encoding']['label_encoding_columns']}")
    print(
        f"      Scale: {final_plan['preprocessing']['scaling']['strategy']} -> "
        f"{len(final_plan['preprocessing']['scaling']['columns'])} columns"
    )

    return final_plan


def main():
    print("\n" + "=" * 70)
    print("PLAN DISPATCHER - TRANSFORMER FIRST WITH PLAN GENERATOR FALLBACK")
    print("=" * 70)
    print(f"Schema dir: {SCHEMA_DIR}")
    print(f"Plan dir:   {PLAN_DIR}")
    print(f"Debug dir:  {DEBUG_DIR}")
    print(f"LoRA dir:   {LORA_DIR}")
    print("=" * 70 + "\n")

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_DIR.exists():
        print("data/schema folder not found.")
        raise SystemExit(1)

    schema_files = sorted(SCHEMA_DIR.glob("*_schema.json"))

    if not schema_files:
        print("No schema files found in data/schema/")
        raise SystemExit(1)

    print(f"Found {len(schema_files)} schema files.\n")

    if model_available():
        print("Model files detected. Dispatcher will try transformer inference first.")
    else:
        print("Model is not available. Dispatcher will use fallback path.")
        print("To enable model inference:")
        print("  1. Put LoRA weights in models/deepseek_stage2/")
        print("  2. Install transformers, peft, accelerate, sentencepiece, and torch")
    print()

    success = 0
    failed = 0

    for i, schema_file in enumerate(schema_files, 1):
        print(f"{i}. Processing: {schema_file.name}")

        try:
            dispatch_schema(schema_file)
            success += 1
            print()
        except Exception as e:
            failed += 1
            print(f"   Failed to dispatch plan for {schema_file.name}: {e}\n")

    print("=" * 70)
    print("PLAN DISPATCHING COMPLETE")
    print("=" * 70)
    print(f"   Successful: {success}")
    print(f"   Failed:     {failed}")
    print(f"Plans saved to: {PLAN_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()