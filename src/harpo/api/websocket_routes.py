"""
HARPO WebSocket + SSE Routes

Delegates to HarpoStreamServer.get_router().
Mount this alongside trajectory_routes.py to add live streaming.

Usage
-----
from harpo.api.websocket_routes import create_streaming_router
from harpo.observability.streaming import HarpoStreamServer

stream_server = HarpoStreamServer()
app.include_router(create_streaming_router(stream_server), prefix="/v1")
"""

from __future__ import annotations

from typing import Optional, Any


def create_streaming_router(stream_server: Optional[Any] = None) -> Any:
    """
    Return a FastAPI router with WebSocket + SSE endpoints.

    Parameters
    ----------
    stream_server : HarpoStreamServer | None
        If None, a new HarpoStreamServer is created with a default
        ObservabilityBridge.
    """
    try:
        from fastapi import APIRouter
    except ImportError as e:
        raise ImportError("fastapi is required for streaming routes.") from e

    from harpo.observability.streaming import HarpoStreamServer

    server = stream_server or HarpoStreamServer()
    router = server.get_router()

    # Additional snapshot endpoint not on HarpoStreamServer
    extra = APIRouter()

    @extra.get("/ws/trajectories/{trajectory_id}/snapshot")
    async def snapshot(trajectory_id: str):
        """Return current metric snapshot without opening a stream."""
        return server.snapshot(trajectory_id)

    # Merge both routers
    combined = APIRouter()
    combined.include_router(router)
    combined.include_router(extra)
    return combined
