"""Tests for the periodic retrain loop."""
# pylint: disable=protected-access

import asyncio
from typing import Any

import pytest

from speech_to_phrase import __main__ as stp_main
from speech_to_phrase.const import Settings, State


def _make_state(tmp_path: Any) -> State:
    settings = Settings(
        models_dir=tmp_path / "models",
        train_dir=tmp_path / "train",
        tools_dir=tmp_path / "tools",
        custom_sentences_dirs=[],
        hass_token="fake-token",
        hass_websocket_uri="ws://localhost:8123/api/websocket",
        retrain_on_connect=False,
    )
    return State(settings=settings)


@pytest.mark.asyncio
async def test_retrain_loop_survives_error(monkeypatch, tmp_path):
    """The retrain loop must keep running after a failed pass.

    _retrain_once() raises when HA is unreachable or auth fails. Without
    exception handling, the loop task died on the first error and no
    retraining ever ran again until the server was restarted.
    """
    calls: list = []
    enough_calls = asyncio.Event()

    async def failing_retrain_once(state: Any, force_retrain: bool = False) -> None:
        calls.append(1)
        if len(calls) >= 3:
            enough_calls.set()
        raise RuntimeError("Home Assistant is down")

    monkeypatch.setattr(stp_main, "_retrain_once", failing_retrain_once)
    state = _make_state(tmp_path)

    loop_task = asyncio.create_task(stp_main._retrain_loop(state, 0.01))
    try:
        # Several passes must happen despite every one of them failing
        await asyncio.wait_for(enough_calls.wait(), timeout=5)
        assert not loop_task.done(), "retrain loop task exited early"
    except asyncio.TimeoutError:
        pytest.fail(
            f"expected at least 3 retrain passes, got {len(calls)}; "
            "the loop died on the first exception"
        )
    finally:
        loop_task.cancel()
        await asyncio.wait([loop_task])
