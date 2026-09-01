"""Async image jobs in the SDK: submit+poll inside generate(), the exposed
primitives, deadline/failure semantics, and old-gateway sync fallback."""

from __future__ import annotations

import pytest

import mlpal_assistants.services.images as images_mod
from mlpal_assistants.exceptions import (
    ImageJobFailedError,
    ImageJobTimeoutError,
)
from mlpal_assistants.exceptions import (
    TimeoutError as SDKTimeoutError,
)
from mlpal_assistants.services.images import ImageService

SYNC_PAYLOAD = {
    "data": [{"url": "https://s3/img.png", "expires_at": "2026-09-01T00:00:00Z"}],
    "cost": {"model_name": "gpt-image-2", "provider": "openai", "compute_units": 0.0159},
}
JOB_QUEUED = {"id": "imgjob_abc", "status": "queued", "created_at": "2026-09-01T00:00:00Z"}


class FakeTransport:
    def __init__(self, post_result, get_results=()):
        self.post_result = post_result
        self.get_results = list(get_results)
        self.posts = []
        self.gets = []

    async def post(self, path, *, body=None, timeout=None):
        self.posts.append((path, body))
        return self.post_result

    async def get(self, path, **kw):
        self.gets.append(path)
        return self.get_results.pop(0)


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    monkeypatch.setattr(images_mod, "_FIRST_POLL_S", 0.001)
    monkeypatch.setattr(images_mod, "_POLL_INTERVAL_S", 0.001)


@pytest.mark.asyncio
async def test_generate_submits_background_and_polls_to_success():
    t = FakeTransport(
        JOB_QUEUED,
        get_results=[
            {"id": "imgjob_abc", "status": "running"},
            {"id": "imgjob_abc", "status": "succeeded", "result": SYNC_PAYLOAD},
        ],
    )
    svc = ImageService(t)
    resp = await svc.generate("gpt-image-2", "a cat")

    path, body = t.posts[0]
    assert path == "/v1/images/generations"
    assert body["wait"] is False
    assert body["idempotency_key"].startswith("sdk-")
    assert t.gets == ["/v1/images/jobs/imgjob_abc"] * 2
    assert resp.images[0].url == "https://s3/img.png"


@pytest.mark.asyncio
async def test_generate_sync_fallback_for_old_gateway():
    t = FakeTransport(SYNC_PAYLOAD)
    resp = await ImageService(t).generate("gpt-image-2", "a cat")
    assert resp.images[0].url == "https://s3/img.png"
    assert t.gets == []  # no polling on a sync answer


@pytest.mark.asyncio
async def test_generate_failed_job_raises_with_code():
    t = FakeTransport(
        JOB_QUEUED,
        get_results=[{
            "id": "imgjob_abc", "status": "failed",
            "error": {"code": "worker_lost", "message": "pod restarted"},
        }],
    )
    with pytest.raises(ImageJobFailedError) as ei:
        await ImageService(t).generate("gpt-image-2", "a cat")
    assert ei.value.code == "worker_lost"
    assert ei.value.job_id == "imgjob_abc"


@pytest.mark.asyncio
async def test_generate_deadline_raises_timeout_with_job_id():
    t = FakeTransport(
        JOB_QUEUED,
        get_results=[{"id": "imgjob_abc", "status": "running"}] * 50,
    )
    with pytest.raises(ImageJobTimeoutError) as ei:
        await ImageService(t).generate("gpt-image-2", "a cat", deadline=0.01)
    assert ei.value.job_id == "imgjob_abc"
    # The MCP's error mapping keys on the SDK's TimeoutError class.
    assert isinstance(ei.value, SDKTimeoutError)


@pytest.mark.asyncio
async def test_primitives_submit_and_get():
    t = FakeTransport(JOB_QUEUED, get_results=[{"id": "imgjob_abc", "status": "running"}])
    svc = ImageService(t)
    job = await svc.submit_job("gpt-image-2", "a cat", idempotency_key="my-key")
    assert job.id == "imgjob_abc" and job.status == "queued" and not job.done
    assert t.posts[0][1]["idempotency_key"] == "my-key"
    polled = await svc.get_job(job.id)
    assert polled.status == "running"
