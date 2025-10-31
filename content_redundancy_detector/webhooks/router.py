from fastapi import APIRouter, Depends, BackgroundTasks, Header, HTTPException
from typing import Optional
from .models import WebhookEvent
from .dispatcher import get_webhook_dispatcher
from .config import settings
import time


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.post("/ingest", status_code=202)
async def ingest_webhook(
    event: WebhookEvent,
    background_tasks: BackgroundTasks,
    x_signature: Optional[str] = Header(default=None),
    x_timestamp: Optional[str] = Header(default=None),
    idempotency_key: Optional[str] = Header(default=None)
):
    # Inject header signature if provided
    if x_signature and not event.signature:
        event.signature = x_signature
    # Attach timestamp if provided
    if x_timestamp and not event.timestamp:
        event.timestamp = x_timestamp

    dispatcher = await get_webhook_dispatcher()

    # Backpressure: if queue near capacity, reject with Retry-After
    qsize = dispatcher.queue.qsize()
    capacity = dispatcher.queue.maxsize or 1
    if qsize / capacity >= settings.WEBHOOK_BACKPRESSURE_THRESHOLD:
        raise HTTPException(status_code=429, detail="Queue busy", headers={
            "Retry-After": str(int(settings.WEBHOOK_RETRY_AFTER_SECS))
        })

    # Validate timestamp window if required
    if settings.WEBHOOK_REQUIRE_TIMESTAMP:
        if not event.timestamp:
            raise HTTPException(status_code=400, detail="Missing X-Timestamp")
        try:
            ts = float(event.timestamp)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid X-Timestamp")
        now = time.time()
        if abs(now - ts) > (settings.WEBHOOK_HMAC_WINDOW_SECS + settings.WEBHOOK_CLOCK_SKEW_SECS):
            raise HTTPException(status_code=401, detail="Timestamp expired")

    # 202 immediately; enqueue in background
    background_tasks.add_task(dispatcher.enqueue, event, idempotency_key)
    return {"accepted": True, "id": event.id}


@router.get("/stats")
async def webhook_stats():
    dispatcher = await get_webhook_dispatcher()
    return {
        "queue_size": dispatcher.queue.qsize(),
        "dlq_size": len(dispatcher.dlq),
        "workers": len(dispatcher.workers),
        "processed_ok": dispatcher.processed_ok,
        "processed_error": dispatcher.processed_error
    }


