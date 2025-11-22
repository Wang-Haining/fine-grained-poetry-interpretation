"""
async guarded backend for an openai-compatible vllm server.

features:
- async httpx client with keepalive
- optional structured output via json schema
- pydantic validation with guided retries
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Type

import httpx
from guardrails import Guard
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)


class GuardedBackend:
    """openai-style async chat completion client with pydantic guardrail retries."""

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

        env_pool = os.getenv("HTTP_POOL")
        env_keep = os.getenv("HTTP_KEEPALIVE")
        env_pool_timeout = os.getenv("HTTP_POOL_TIMEOUT")

        self.http_pool = int(env_pool) if env_pool else int(http_pool)
        self.keepalive_expiry = float(env_keep) if env_keep else 120.0
        self.pool_timeout = (
            float(env_pool_timeout) if env_pool_timeout else self.read_timeout
        )

        self.client_lock = asyncio.Lock()
        self.client = self.create_http_client()
        self.guards: Dict[Type[BaseModel], Guard] = {}

    def create_http_client(self) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=self.http_pool,
            max_keepalive_connections=self.http_pool,
            keepalive_expiry=self.keepalive_expiry,
        )
        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.read_timeout,
            pool=self.pool_timeout,
        )
        transport = httpx.AsyncHTTPTransport(limits=limits, retries=3)
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers={"Connection": "keep-alive"},
        )

    async def ensure_client_open(self) -> httpx.AsyncClient:
        if self.client is not None and not self.client.is_closed:
            return self.client
        async with self.client_lock:
            if self.client is None or self.client.is_closed:
                self.client = self.create_http_client()
            return self.client

    async def post_json(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        delays = [0.25, 0.5, 1.0, 2.0]
        for step, delay in enumerate([0.0] + delays):
            if delay:
                await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
            client = await self.ensure_client_open()
            try:
                return await client.post(path, json=payload)
            except RuntimeError as exc:
                if "closed" in str(exc).lower() and step < len(delays):
                    continue
                raise
            except (httpx.PoolTimeout, httpx.ReadError):
                if step == len(delays):
                    raise
                continue

    async def close(self) -> None:
        async with self.client_lock:
            if self.client is not None and not self.client.is_closed:
                await self.client.aclose()
            self.client = None

    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        json_schema: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }

        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        if reasoning_effort is not None:
            payload["reasoning_effort"] = str(reasoning_effort)

        resp = await self.post_json("/v1/chat/completions", payload)

        if resp.status_code == 500 and json_schema is not None:
            # simple fallback: relax strict, then plain json_object
            try_payload = dict(payload)
            schema_rf = dict(try_payload.get("response_format", {}))
            schema_block = dict(schema_rf.get("json_schema", {}))
            if schema_block.get("strict", True):
                schema_block["strict"] = False
                schema_rf["json_schema"] = schema_block
                try_payload["response_format"] = schema_rf
                resp2 = await self.post_json("/v1/chat/completions", try_payload)
                if resp2.status_code < 400:
                    return resp2.json()

            try_payload = dict(payload)
            try_payload["response_format"] = {"type": "json_object"}
            resp3 = await self.post_json("/v1/chat/completions", try_payload)
            if resp3.status_code < 400:
                return resp3.json()

        if resp.status_code >= 400:
            logger.error(resp.text[:2000])
        resp.raise_for_status()
        return resp.json()

    async def guardrail(
        *,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        json_schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 6,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_effort: str = "high",
        max_tokens: int = 256,
    ) -> BaseModel:
        guard = self.guards.get(response_model)
        if guard is None:
            guard = Guard.from_pydantic(response_model)
            self.guards[response_model] = guard

        def try_extract_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
            if not raw:
                return None
            # 1) direct parse
            try:
                return json.loads(raw)
            except Exception:
                pass
            # 2) naive slice between first '{' and last '}'
            s = raw.find("{")
            e = raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                sub = raw[s : e + 1]
                try:
                    return json.loads(sub)
                except Exception:
                    return None
            return None

        current_messages = list(messages)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            # main call (schema on)
            try:
                resp = await self.generate(
                    messages=current_messages,
                    temperature=temperature,
                    top_p=top_p,
                    json_schema=json_schema,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                # backoff + continue
                await asyncio.sleep(0.25 * (attempt + 1))
                continue

            # unpack
            try:
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("parsed")
                raw = msg.get("content")
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.25 * (attempt + 1))
                continue

            # if we got nothing, immediately **re-call** once with schema OFF + coercive hint
            if content is None and not raw and json_schema is not None:
                fallback_messages = list(current_messages) + [
                    {
                        "role": "system",
                        "content": (
                            "output ONLY one JSON object that matches the schema. "
                            "no markdown, no prose, no code fences. "
                            'if unsure, return {"emotions":["sadness"],'
                            '"sentiment":"neutral","themes":["others"]}.'
                        ),
                    }
                ]
                try:
                    resp2 = await self.generate(
                        messages=fallback_messages,
                        temperature=temperature,
                        top_p=top_p,
                        json_schema=None,  # schema OFF here
                        reasoning_effort=reasoning_effort,
                        max_tokens=max_tokens,
                    )
                    choice2 = (resp2.get("choices") or [{}])[0]
                    msg2 = choice2.get("message") or {}
                    content = msg2.get("parsed")
                    raw = msg2.get("content")
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue

            # parse content
            if content is None:
                # try to parse raw
                parsed = try_extract_json(raw)
                if parsed is None:
                    # re-ask with a short, coercive hint
                    last_error = ValueError("no json content returned")
                    current_messages = list(messages) + [
                        {
                            "role": "system",
                            "content": (
                                "your previous reply did not contain valid json. "
                                "respond again with only a single json object, no markdown."
                            ),
                        }
                    ]
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue
                content = parsed

            if not isinstance(content, dict):
                last_error = ValueError("non json object returned")
                current_messages = list(messages) + [
                    {
                        "role": "system",
                        "content": (
                            "respond again with a single json object that matches the schema. "
                            "no arrays, no prose."
                        ),
                    }
                ]
                await asyncio.sleep(0.25 * (attempt + 1))
                continue

            # validate via pydantic
            try:
                validated = response_model.model_validate(content)
                return validated
            except ValidationError as exc:
                last_error = exc
                hint = (
                    "validation failed. fix the json to satisfy the schema exactly. "
                    f"errors: {str(exc)}. output json only."
                )
                current_messages = list(messages) + [
                    {"role": "system", "content": hint}
                ]
                await asyncio.sleep(0.25 * (attempt + 1))
                continue

        assert last_error is not None
        raise last_error
