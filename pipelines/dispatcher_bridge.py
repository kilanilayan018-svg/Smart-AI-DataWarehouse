"""
dispatcher_bridge.py
--------------------
Bridge between services.py (which already has the schema as an in-memory dict
and the model API response) and plan_dispatcher.py (which holds the smart
validation + fallback logic).

services.py calls ONE function here and always gets back a fully validated,
normalized plan — whether it came from:
    1. the model API (validated and accepted),
    2. the rule-based PlanGenerator (when the model output is bad/weak), or
    3. an internal safe fallback (when even PlanGenerator fails).

It never returns None, never raises to the caller, and never breaks the
API response contract. If anything unexpected happens it degrades to the
safest available plan.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("smart_ai_dw_api")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def dispatch_from_model_response(
    model_response: Optional[dict],
    schema: dict,
    dataset_name: str,
    target_column: Optional[str] = None,
) -> dict:
    """
    Run the model API response through the dispatcher's validation + fallback chain.

    Args:
        model_response: Plan dict from model_client.generate_plan_with_model,
                        or None if the model API failed / was disabled.
                        Shape from model_client:
                            {
                              "dataset_name": ...,
                              "target_column": ...,
                              "task_type": ...,
                              "preprocessing": <raw model body>,
                              "_meta": {...},
                            }
        schema:        Schema dict (with _meta) from schema_extractor.
        dataset_name:  Dataset name string.
        target_column: Optional user-provided target override.

    Returns:
        A normalized, validated plan dict (the dispatcher's contract). Never None.
    """
    # Import here so a missing optional dependency in plan_dispatcher never
    # breaks module import of this bridge.
    from pipelines.plan_dispatcher import (
        normalize_transformer_plan,
        normalize_plan_generator_output,
        build_internal_safe_fallback,
        validate_final_plan,
        generate_plan_generator_fallback,
    )

    # If the user explicitly named a target column, make sure the schema _meta
    # reflects it so the dispatcher's target selection respects it.
    if target_column:
        if not isinstance(schema.get("_meta"), dict):
            schema["_meta"] = {}
        schema["_meta"]["target_column"] = target_column

    transformer_reasons = []

    # ── Step 1: Try to accept the model response ──────────────────────────────
    if isinstance(model_response, dict):
        try:
            # model_client wraps the raw API body under "preprocessing".
            model_body = model_response.get("preprocessing", {})
            if not isinstance(model_body, dict):
                model_body = {}

            # Build a raw_plan the dispatcher's normalizer understands. The
            # normalizer looks for target_column / task_type / preprocessing.
            raw_plan = {
                "target_column": (
                    model_response.get("target_column")
                    or model_body.get("target_column")
                ),
                "task_type": (
                    model_response.get("task_type")
                    or model_body.get("task_type")
                ),
                # The model body itself is the preprocessing description.
                "preprocessing": model_body,
            }

            transformer_plan = normalize_transformer_plan(
                raw_plan=raw_plan,
                schema=schema,
                dataset_name=dataset_name,
                parse_status="api_response",
                raw_output_path=None,
            )

            transformer_valid, transformer_reasons = validate_final_plan(
                transformer_plan,
                schema,
                strict_transformer=False,  # API response already structured; don't over-reject
            )

            if transformer_valid:
                transformer_plan.setdefault("_debug", {})["final_decision"] = "accepted_transformer_plan"
                log.info("[dispatcher_bridge] model plan accepted")
                return transformer_plan

            log.warning(
                "[dispatcher_bridge] model plan rejected: %s",
                "; ".join(transformer_reasons) or "unknown",
            )

        except Exception as e:
            log.warning("[dispatcher_bridge] error normalizing model response: %s", e)
            transformer_reasons = [f"normalize_error: {e}"]
    else:
        log.info("[dispatcher_bridge] no model response — going to PlanGenerator fallback")
        transformer_reasons = ["model_returned_none"]

    fallback_reason = "; ".join(transformer_reasons) or "model_unavailable_or_invalid"

    # ── Step 2: PlanGenerator fallback ────────────────────────────────────────
    log.info("[dispatcher_bridge] trying PlanGenerator fallback (%s)", fallback_reason)
    try:
        legacy_plan, legacy_error = generate_plan_generator_fallback(schema, dataset_name)
    except Exception as e:
        legacy_plan, legacy_error = None, str(e)

    if legacy_plan is not None:
        try:
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
                fallback_plan.setdefault("_debug", {})["final_decision"] = "used_plan_generator_fallback"
                fallback_plan["_debug"]["transformer_rejection_reasons"] = transformer_reasons
                log.info("[dispatcher_bridge] PlanGenerator fallback accepted")
                return fallback_plan

            log.warning(
                "[dispatcher_bridge] PlanGenerator fallback invalid: %s",
                "; ".join(fallback_reasons) or "unknown",
            )
            emergency_reason = f"plan_generator_invalid: {'; '.join(fallback_reasons)}"

        except Exception as e:
            log.warning("[dispatcher_bridge] error normalizing PlanGenerator output: %s", e)
            emergency_reason = f"plan_generator_normalize_error: {e}"
    else:
        emergency_reason = f"plan_generator_failed: {legacy_error}"
        log.warning("[dispatcher_bridge] %s", emergency_reason)

    # ── Step 3: Internal safe fallback — always succeeds ──────────────────────
    log.info("[dispatcher_bridge] using internal safe fallback")
    safe_plan = build_internal_safe_fallback(
        schema=schema,
        dataset_name=dataset_name,
        fallback_reason=emergency_reason,
    )
    safe_plan.setdefault("_debug", {})["final_decision"] = "used_internal_safe_fallback"
    safe_plan["_debug"]["transformer_rejection_reasons"] = transformer_reasons
    return safe_plan
