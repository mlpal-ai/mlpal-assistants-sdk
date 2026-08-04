"""`client.feedback` — outcome signal at `/v2/feedback`.

Report how a delegated subtask turned out. These signals refine the curated
catalog's quality rankings over time, so a harness that routes by
`catalog.tiers` closes the loop by reporting outcomes here. Unknown model tags
or task types are stored as-is (the gateway is deliberately lenient).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mlpal_assistants._wire import FeedbackResult

if TYPE_CHECKING:
    from mlpal_assistants._transport import AsyncTransport, SyncTransport

_PATH = "/v2/feedback"

# Mirrors the gateway's FEEDBACK_OUTCOMES; kept as a Literal for editor hints
# while the transport still forwards any string (the gateway stays lenient).
Outcome = Literal["accepted", "failed", "escalated", "retried"]


def _body(
    model: str,
    task_type: str,
    outcome: str,
    escalated_to: str | None,
    trace_id: str | None,
) -> dict[str, object]:
    body: dict[str, object] = {"model": model, "task_type": task_type, "outcome": outcome}
    if escalated_to is not None:
        body["escalated_to"] = escalated_to
    if trace_id is not None:
        body["trace_id"] = trace_id
    return body


class Feedback:
    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        model: str,
        task_type: str,
        outcome: Outcome,
        escalated_to: str | None = None,
        trace_id: str | None = None,
    ) -> FeedbackResult:
        data, _ = self._transport.request(
            "POST", _PATH, json=_body(model, task_type, outcome, escalated_to, trace_id)
        )
        return FeedbackResult.model_validate(data)


class AsyncFeedback:
    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        model: str,
        task_type: str,
        outcome: Outcome,
        escalated_to: str | None = None,
        trace_id: str | None = None,
    ) -> FeedbackResult:
        data, _ = await self._transport.request(
            "POST", _PATH, json=_body(model, task_type, outcome, escalated_to, trace_id)
        )
        return FeedbackResult.model_validate(data)
