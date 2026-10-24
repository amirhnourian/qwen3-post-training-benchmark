"""Tests use local fixtures; network access is never an implicit dependency."""

import socket

import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is disabled in the test suite")

    for method in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, method, blocked)
    for method in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, method, blocked)
