from __future__ import annotations

import io
import json
import os
import socket
from typing import Any

import pytest

from gs_video import app as local_app


def test_bind_api_socket_reserves_random_loopback_port() -> None:
    listener = local_app.bind_api_socket("127.0.0.1")
    try:
        host, port = listener.getsockname()
        assert host == "127.0.0.1"
        assert 1 <= port <= 65_535
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        listener.close()


def test_bind_api_socket_rejects_non_loopback_host() -> None:
    with pytest.raises(ValueError, match="loopback"):
        local_app.bind_api_socket("0.0.0.0")


def test_run_server_emits_one_bounded_handshake_and_uses_reserved_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = io.StringIO()
    token = "must-not-leak"
    calls: list[list[socket.socket] | None] = []

    class StubServer:
        should_exit = False

        def run(self, *, sockets: list[socket.socket] | None = None) -> None:
            calls.append(sockets)

    listener = local_app.bind_api_socket("127.0.0.1")
    port = int(listener.getsockname()[1])
    monkeypatch.setattr(local_app, "bind_api_socket", lambda host: listener)
    app = type("StubApp", (), {"state": type("State", (), {})()})()

    local_app.run_server(
        app,
        bind_host="127.0.0.1",
        port=0,
        startup_handshake=True,
        output=output,
        server_factory=lambda config: StubServer(),
        config_factory=lambda *args, **kwargs: object(),
    )

    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    handshake = json.loads(lines[0])
    assert handshake == {
        "port": port,
        "apiVersion": "v1",
        "pid": os.getpid(),
        "parentPid": os.getppid(),
    }
    assert token not in output.getvalue()
    assert calls == [[listener]]
    assert listener.fileno() == -1


def test_run_server_without_handshake_uses_configured_bind() -> None:
    calls: list[list[socket.socket] | None] = []
    config_arguments: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class StubServer:
        should_exit = False

        def run(self, *, sockets: list[socket.socket] | None = None) -> None:
            calls.append(sockets)

    app = type("StubApp", (), {"state": type("State", (), {})()})()

    local_app.run_server(
        app,
        bind_host="127.0.0.1",
        port=43210,
        startup_handshake=False,
        server_factory=lambda config: StubServer(),
        config_factory=lambda *args, **kwargs: (
            config_arguments.append((args, kwargs)) or object()
        ),
    )

    assert calls == [None]
    assert config_arguments[0][1]["host"] == "127.0.0.1"
    assert config_arguments[0][1]["port"] == 43210
    assert callable(app.state.request_shutdown)
