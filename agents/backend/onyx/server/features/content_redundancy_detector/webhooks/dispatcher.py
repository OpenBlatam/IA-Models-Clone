import asyncio
import hmac
import hashlib
import json
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import aiohttp
import orjson

from .config import settings
from .models import WebhookEvent, WebhookResult

# Reuse resilience utilities
from ..infrastructure.resilience import (
    RetryPolicy,
    RetryStrategy,
    retry_async,
)


@dataclass
class DLQItem:
    event: WebhookEvent
    error: str
    attempts: int
    last_attempt: float


class WebhookDispatcher:
    def __init__(self):
        self.queue: asyncio.Queue[WebhookEvent] = asyncio.Queue(maxsize=settings.WEBHOOK_QUEUE_SIZE)
        self.dlq: List[DLQItem] = []
        self.workers: List[asyncio.Task] = []
        self.running: bool = False
        self.handler: Optional[Callable[[WebhookEvent], Any]] = None
        # Idempotency store (event.id -> expire_at)
        self._seen: Dict[str, float] = {}
        # Simple stats
        self.processed_ok: int = 0
        self.processed_error: int = 0
        # HTTP client pool
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def start(self, handler: Callable[[WebhookEvent], Any]):
        if self.running:
            return
        self.running = True
        self.handler = handler
        
        # Initialize HTTP session
        connector = aiohttp.TCPConnector(
            limit=settings.WEBHOOK_HTTP_POOL_SIZE,
            limit_per_host=settings.WEBHOOK_HTTP_POOL_SIZE // 4,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=settings.WEBHOOK_HTTP_TIMEOUT)
        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=orjson.dumps
        )
        
        # Start workers
        for i in range(settings.WEBHOOK_MAX_WORKERS):
            self.workers.append(asyncio.create_task(self._worker_loop(f"wh-worker-{i}")))
        
        # Prewarm workers if enabled
        if settings.WEBHOOK_PREWARM_WORKERS:
            await self._prewarm_workers()

    async def stop(self):
        self.running = False
        for w in self.workers:
            w.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def enqueue(self, event: WebhookEvent, idempotency_key: Optional[str] = None):
        # Idempotency: drop if seen within TTL
        now = time.time()
        key = idempotency_key or event.id
        expire = self._seen.get(key)
        if expire and expire > now:
            return
        # record seen
        self._seen[key] = now + settings.WEBHOOK_IDEMPOTENCY_TTL_SECS
        # cleanup occasionally (cheap)
        if len(self._seen) % 1000 == 0:
            cutoff = time.time()
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
        await self.queue.put(event)

    async def _worker_loop(self, worker_id: str):
        batch: List[WebhookEvent] = []
        batch_deadline = time.time() + settings.WEBHOOK_BATCH_WAIT_SECS
        retry_policy = RetryPolicy(
            max_attempts=settings.WEBHOOK_RETRY_MAX_ATTEMPTS,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=settings.WEBHOOK_RETRY_INITIAL_DELAY,
            max_delay=settings.WEBHOOK_RETRY_MAX_DELAY,
            jitter=True,
        )
        while self.running:
            try:
                timeout = max(0.01, batch_deadline - time.time())
                try:
                    evt: WebhookEvent = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(evt)
                except asyncio.TimeoutError:
                    pass

                # Adaptive batch: scale with queue pressure
                target_batch = settings.WEBHOOK_BATCH_SIZE
                qsize = self.queue.qsize()
                capacity = self.queue.maxsize or 1
                pressure = qsize / capacity
                if pressure > 0.75:
                    target_batch = min(settings.WEBHOOK_BATCH_SIZE * 2, 4 * settings.WEBHOOK_BATCH_SIZE)
                elif pressure < 0.25:
                    target_batch = max(10, settings.WEBHOOK_BATCH_SIZE // 2)

                if len(batch) >= target_batch or time.time() >= batch_deadline:
                    if batch and self.handler:
                        events = batch
                        batch = []
                        batch_deadline = time.time() + settings.WEBHOOK_BATCH_WAIT_SECS
                        # Process batch concurrently
                        results = await asyncio.gather(*[
                            self._process_event(e, retry_policy) for e in events
                        ], return_exceptions=True)
                        # update stats
                        for r in results:
                            if isinstance(r, WebhookResult) and r.success:
                                self.processed_ok += 1
                            else:
                                self.processed_error += 1
                    else:
                        batch_deadline = time.time() + settings.WEBHOOK_BATCH_WAIT_SECS
            except asyncio.CancelledError:
                break
            except Exception:
                # Keep loop healthy on unexpected errors
                continue

    async def _process_event(self, event: WebhookEvent, retry_policy: RetryPolicy) -> WebhookResult:
        async def handle_once(e: WebhookEvent):
            # Verify signature if configured
            if settings.WEBHOOK_HMAC_SECRET:
                self._verify_signature(e)
            # Call user handler (can be async or sync)
            if asyncio.iscoroutinefunction(self.handler):
                return await self.handler(e)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: self.handler(e))

        try:
            await retry_async(handle_once, event, policy=retry_policy)
            return WebhookResult(id=event.id, success=True, attempts=1)
        except Exception as err:
            # To DLQ after retries exhausted
            self.dlq.append(DLQItem(event=event, error=str(err), attempts=retry_policy.max_attempts, last_attempt=time.time()))
            return WebhookResult(id=event.id, success=False, attempts=retry_policy.max_attempts, error=str(err))

    def _verify_signature(self, event: WebhookEvent):
        # Signature over payload + '.' + timestamp (if provided)
        raw = orjson.dumps(event.payload, option=orjson.OPT_SORT_KEYS)
        if settings.WEBHOOK_REQUIRE_TIMESTAMP and event.timestamp:
            message = raw + b'.' + event.timestamp.encode()
        else:
            message = raw
        expected = hmac.new(settings.WEBHOOK_HMAC_SECRET.encode(), message, hashlib.sha256).hexdigest()
        if not event.signature or not hmac.compare_digest(expected, event.signature):
            raise ValueError("Invalid webhook signature")

    async def _prewarm_workers(self):
        """Prewarm workers with a dummy task to initialize them"""
        dummy_event = WebhookEvent(id="prewarm", type="prewarm", payload={})
        await self.queue.put(dummy_event)
        # Wait a bit for workers to process
        await asyncio.sleep(0.1)


# Singleton accessor
_dispatcher: Optional[WebhookDispatcher] = None


async def get_webhook_dispatcher() -> WebhookDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WebhookDispatcher()
    return _dispatcher


