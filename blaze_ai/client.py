from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class BlazeAIClient:
    def __init__(
        self,
        base_url: str,
        api_prefix: str = "/api",
        timeout: float = 15.0,
        headers: Optional[Dict[str, str]] = None,
        concurrency: int = 8,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._prefix = "/" + api_prefix.strip("/") if api_prefix else ""
        self._timeout = float(timeout)
        self._headers = headers or {}
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers, http2=True)
        self._sem = asyncio.Semaphore(int(concurrency))

    def _url(self, path: str) -> str:
        return f"{self._base}{self._prefix}{path}"

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- Core ---
    async def health(self) -> Dict[str, Any]:
        async with self._sem:
            r = await self._client.get(self._url("/blaze/health"))
        r.raise_for_status()
        return r.json()

    async def stats(self) -> Dict[str, Any]:
        async with self._sem:
            r = await self._client.get(self._url("/blaze/stats"))
        r.raise_for_status()
        return r.json()

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._post_json("/blaze/process", payload)

    async def process_batch(self, payloads: List[Dict[str, Any]], concurrency: int = 8) -> Dict[str, Any]:
        body = {"payloads": payloads, "concurrency": concurrency}
        return await self._post_json("/blaze/process/batch", body)

    # --- Feature helpers ---
    async def brand_train(self, brand_name: str, samples: List[str]) -> Dict[str, Any]:
        body = {"brand_name": brand_name, "samples": samples}
        return await self._post_json("/blaze/brand/train", body)

    async def post_social(self, topic: str, brand_name: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        body = {"topic": topic, "brand_name": brand_name, "timeout": timeout}
        return await self._post_json("/blaze/post/social", body)

    async def email_create(self, subject: str, points: List[str], brand_name: str) -> Dict[str, Any]:
        body = {"subject": subject, "points": points, "brand_name": brand_name}
        return await self._post_json("/blaze/email/create", body)

    async def blog_outline(self, title: str, sections: List[str], brand_name: str) -> Dict[str, Any]:
        body = {"title": title, "sections": sections, "brand_name": brand_name}
        return await self._post_json("/blaze/blog/outline", body)

    async def seo_meta(self, title: str, summary: str, topic: Optional[str] = None, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        body = {"title": title, "summary": summary, "topic": topic, "keywords": keywords}
        return await self._post_json("/blaze/seo/meta", body)

    async def llm_generate(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = {"prompt": prompt, "overrides": overrides or {}}
        return await self._post_json("/blaze/llm/generate", body)

    async def diffusion_generate(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }
        return await self._post_json("/blaze/diffusion/generate", body)

    # --- Planner ---
    async def planner_create(self, topic: str, channels: List[str], cadence_days: int, num_posts: int) -> Dict[str, Any]:
        body = {"topic": topic, "channels": channels, "cadence_days": cadence_days, "num_posts": num_posts}
        return await self._post_json("/blaze/planner/create", body)

    async def planner_list(self) -> Dict[str, Any]:
        async with self._sem:
            r = await self._client.get(self._url("/blaze/planner/list"))
        r.raise_for_status()
        return r.json()

    async def planner_schedule(self, channel: str, content: Dict[str, Any], scheduled_at_iso: str, plan_id: Optional[str] = None) -> Dict[str, Any]:
        body = {"channel": channel, "content": content, "scheduled_at": scheduled_at_iso, "plan_id": plan_id}
        return await self._post_json("/blaze/planner/schedule", body)

    async def planner_scheduled(self, status: Optional[str] = None) -> Dict[str, Any]:
        params = {"status": status} if status else None
        async with self._sem:
            r = await self._client.get(self._url("/blaze/planner/scheduled"), params=params)
        r.raise_for_status()
        return r.json()

    async def planner_cancel(self, item_id: str) -> Dict[str, Any]:
        async with self._sem:
            r = await self._client.post(self._url("/blaze/planner/cancel"), params={"item_id": item_id})
        r.raise_for_status()
        return r.json()

    # --- internals ---
    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.2, min=0.2, max=1.5), retry=retry_if_exception_type(httpx.HTTPError))
    async def _post_json(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        async with self._sem:
            r = await self._client.post(self._url(path), json=body)
        r.raise_for_status()
        return r.json()


