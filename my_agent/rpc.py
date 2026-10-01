from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4


RPC_VERSION = 1


@dataclass(frozen=True)
class RequestEnvelope:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class RpcError:
    code: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseEnvelope:
    request_id: str | None
    result: dict[str, Any] | None = None
    error: RpcError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("RPC response must contain exactly one of result or error")

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(
        cls,
        request_id: str,
        result: Mapping[str, Any] | None = None,
    ) -> ResponseEnvelope:
        return cls(request_id=request_id, result=dict(result or {}))

    @classmethod
    def failure(
        cls,
        request_id: str | None,
        *,
        code: str,
        message: str,
        data: Mapping[str, Any] | None = None,
    ) -> ResponseEnvelope:
        return cls(
            request_id=request_id,
            error=RpcError(code=code, message=message, data=dict(data or {})),
        )


def request_to_dict(request: RequestEnvelope) -> dict[str, Any]:
    return {
        "channel": "request",
        "rpc_version": RPC_VERSION,
        "request_id": request.request_id,
        "method": request.method,
        "params": request.params,
    }


def request_from_dict(data: Mapping[str, Any]) -> RequestEnvelope:
    _require_version(data)
    if data.get("channel") != "request":
        raise ValueError("RPC channel must be 'request'")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("RPC request_id must be a non-empty string")
    method = data.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("RPC method must be a non-empty string")
    params = data.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("RPC params must be an object")
    return RequestEnvelope(
        request_id=request_id,
        method=method,
        params=dict(params),
    )


def response_to_dict(response: ResponseEnvelope) -> dict[str, Any]:
    data: dict[str, Any] = {
        "channel": "response",
        "rpc_version": RPC_VERSION,
        "request_id": response.request_id,
        "ok": response.ok,
    }
    if response.error is None:
        data["result"] = response.result
    else:
        data["error"] = {
            "code": response.error.code,
            "message": response.error.message,
            "data": response.error.data,
        }
    return data


def response_from_dict(data: Mapping[str, Any]) -> ResponseEnvelope:
    _require_version(data)
    if data.get("channel") != "response":
        raise ValueError("RPC channel must be 'response'")
    request_id = data.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise ValueError("RPC response request_id must be a string or null")
    ok = data.get("ok")
    if not isinstance(ok, bool):
        raise ValueError("RPC response ok must be a boolean")
    if ok:
        result = data.get("result", {})
        if not isinstance(result, Mapping):
            raise ValueError("RPC result must be an object")
        return ResponseEnvelope(request_id=request_id, result=dict(result))
    error = data.get("error")
    if not isinstance(error, Mapping):
        raise ValueError("RPC error must be an object")
    code = error.get("code")
    message = error.get("message")
    error_data = error.get("data", {})
    if not isinstance(code, str) or not code:
        raise ValueError("RPC error code must be a non-empty string")
    if not isinstance(message, str) or not message:
        raise ValueError("RPC error message must be a non-empty string")
    if not isinstance(error_data, Mapping):
        raise ValueError("RPC error data must be an object")
    return ResponseEnvelope(
        request_id=request_id,
        error=RpcError(code=code, message=message, data=dict(error_data)),
    )


def _require_version(data: Mapping[str, Any]) -> None:
    if data.get("rpc_version") != RPC_VERSION:
        raise ValueError(f"unsupported RPC version: {data.get('rpc_version')}")
