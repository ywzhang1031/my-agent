"""Offline Web smoke server. All sessions and tools live in a temporary workspace."""
from __future__ import annotations

import signal
import tempfile
from pathlib import Path

from my_agent.agent_loop import AgentLoop
from my_agent.app_server import AppServer
from my_agent.cli import build_tool_registry
from my_agent.config import ConfiguredModel
from my_agent.event_store import EventStore
from my_agent.model_invoker import ModelInvoker
from my_agent.modes import COLLABORATION_MODES, get_mode
from my_agent.permissions import PermissionPolicy
from my_agent.provider import ModelMetadata, ModelResponse, ScriptedProviderAdapter
from my_agent.runtime import AgentRuntime
from my_agent.service import AgentService
from my_agent.session import SessionStore
from my_agent.trace import TraceRecorder
from my_agent.web_server import LocalWebServer
from my_agent.workspace import Workspace


def main():
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    with tempfile.TemporaryDirectory(prefix="my-agent-web-test-") as temporary:
        root = Path(temporary)
        store = SessionStore(root / ".my-agent" / "sessions")
        service = AgentService(tool_registry_factory=build_tool_registry, save_session=store.save,
                               event_store=EventStore(store.root))

        def factory(state, model_ref, mode_id, permission_profile_id, sandbox_mode):
            state.provider_id, state.model_id = "scripted", "scripted-model"
            state.mode_id = mode_id or state.mode_id
            mode = get_mode(state.mode_id)
            replies = [ModelResponse.from_parts(text="Offline fixture response", metadata=ModelMetadata(
                provider_id="scripted", model_id="scripted-model")) for _ in range(20)]
            loop = AgentLoop(workspace=Workspace(root), model=ModelInvoker(ScriptedProviderAdapter(replies)),
                             tools=build_tool_registry(mode), permissions=PermissionPolicy(),
                             trace=TraceRecorder(state.trace_path), mode=mode)
            return AgentRuntime(loop, state, interactive_approvals=False)

        app = AppServer(workspace=root, session_store=store, agent_service=service, runtime_factory=factory,
                        models=(ConfiguredModel(provider_id="scripted", model_id="scripted-model",
                                                description="Offline fixture", context_window_tokens=10000,
                                                max_output_tokens=1000),), modes=COLLABORATION_MODES)
        server = LocalWebServer(app, port=0)
        print(server.url, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
