from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class VllmHttpError(Exception):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class VllmClient:
    def __init__(self, *, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    async def __aenter__(self) -> "VllmClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def health(self) -> dict:
        response = await self._http.get("/health")
        if response.status_code >= 400:
            raise VllmHttpError(
                status_code=response.status_code,
                message=response.text,
            )
        if not response.content:
            return {"status": "ok"}
        return response.json()

    async def list_models(self) -> dict:
        return await self._get("/v1/models")

    async def version(self) -> dict:
        return await self._get("/version")

    async def metrics_text(self) -> str:
        response = await self._http.get("/metrics")
        if response.status_code >= 400:
            raise VllmHttpError(
                status_code=response.status_code,
                message=response.text,
            )
        return response.text

    async def chat_completion(self, body: dict) -> dict:
        return await self._post_json("/v1/chat/completions", body)

    async def completion(self, body: dict) -> dict:
        return await self._post_json("/v1/completions", body)

    async def tokenize(self, body: dict) -> dict:
        return await self._post_json("/tokenize", body)

    async def chat_completion_stream(self, body: dict) -> AsyncIterator[dict]:
        async for chunk in self._post_stream("/v1/chat/completions", body):
            yield chunk

    async def _get(self, path: str) -> dict:
        response = await self._http.get(path)
        return self._json_or_raise(response)

    async def _post_json(self, path: str, body: dict) -> dict:
        response = await self._http.post(path, json=body)
        return self._json_or_raise(response)

    async def _post_stream(self, path: str, body: dict) -> AsyncIterator[dict]:
        payload = dict(body)
        payload["stream"] = True
        async with self._http.stream("POST", path, json=payload) as response:
            if response.status_code != 200:
                content = await response.aread()
                raise VllmHttpError(
                    status_code=response.status_code,
                    message=content.decode("utf-8", errors="replace"),
                )
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        yield {"_done": True}
                        return
                    yield json.loads(data)

    @staticmethod
    def _json_or_raise(response: httpx.Response) -> dict:
        if response.status_code >= 400:
            raise VllmHttpError(
                status_code=response.status_code,
                message=response.text,
            )
        return response.json()
