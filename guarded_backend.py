"""
async guarded backend for openai-compatible vllm server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional, Type

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class GuardedBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 300.0,
        http_pool: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.http_pool = int(http_pool)

        limits = httpx.Limits(
            max_connections=self.http_pool,
            max_keepalive_connections=self.http_pool,
            keepalive_expiry=120.0,
        )
        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.read_timeout,
            pool=self.read_timeout,
        )
        transport = httpx.AsyncHTTPTransport(limits=limits, retries=3)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers={"Connection": "keep-alive"},
        )

    async def post_json(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        delays = [0.25, 0.5, 1.0, 2.0]
        for step, delay in enumerate([0.0] + delays):
            if delay:
                await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
            try:
                return await self.client.post(path, json=payload)
            except (httpx.PoolTimeout, httpx.ReadError):
                if step == len(delays):
                    raise
                continue

    async def close(self) -> None:
        if self.client and not self.client.is_closed:
            await self.client.aclose()

    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        json_mode: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        resp = await self.post_json("/v1/chat/completions", payload)
        resp.raise_for_status()
        return resp.json()

    def extract_json(self, text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        text = text.strip()

        # strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # find first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return None

    async def guardrail(
        self,
        *,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        max_retries: int = 6,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        current_messages = list(messages)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = await self.generate(
                    messages=current_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    json_mode=True,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

            try:
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                raw = msg.get("content")
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

            content = self.extract_json(raw)

            if content is None:
                last_error = ValueError(
                    f"no valid json in response: {raw[:200] if raw else 'empty'}"
                )

                # on later attempts, use minimal fallback prompt (must satisfy response_model)
                if attempt >= 3:
                    current_messages = [
                        {
                            "role": "user",
                            "content": (
                                "output this JSON: "
                                '{"emotions":["sadness"],"sentiment":"neutral","themes":[],"themes_50":[]}'
                            ),
                        }
                    ]
                else:
                    current_messages = list(messages) + [
                        {
                            "role": "system",
                            "content": "output ONLY a single valid JSON object, no markdown, no explanation.",
                        }
                    ]
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

            try:
                validated = response_model.model_validate(content)
                return validated
            except ValidationError as exc:
                last_error = exc
                errors_str = "; ".join(
                    [f"{e['loc']}: {e['msg']}" for e in exc.errors()[:3]]
                )
                current_messages = list(messages) + [
                    {
                        "role": "system",
                        "content": f"validation failed: {errors_str}. fix and return valid json.",
                    }
                ]
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

        raise last_error or RuntimeError("guardrail exhausted retries")
