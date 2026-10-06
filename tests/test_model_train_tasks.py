"""Tests for training task bookkeeping in model_train_tasks."""
# pylint: disable=protected-access

import asyncio
from typing import Any, List

import pytest

from speech_to_phrase import __main__ as stp_main
from speech_to_phrase.const import Settings, State
from speech_to_phrase.hass_api import HomeAssistantInfo, Things

FAKE_MODEL_ID = "de_de"


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


def _patch_retrain(monkeypatch: Any, train_model: Any) -> None:
    """Patch HA access and training.

    _retrain_once() resolves get_hass_info, get_models_for_languages and
    _train_model as module globals, so they are patched on __main__.
    """

    async def fake_get_hass_info(token: str, uri: str) -> HomeAssistantInfo:
        return HomeAssistantInfo(
            system_language="de",
            things=Things(),
            pipeline_languages={"de"},
        )

    def fake_get_models_for_languages(languages: Any) -> List[Any]:
        class _FakeModel:
            id = FAKE_MODEL_ID

        return [_FakeModel()]

    monkeypatch.setattr(stp_main, "get_hass_info", fake_get_hass_info)
    monkeypatch.setattr(
        stp_main, "get_models_for_languages", fake_get_models_for_languages
    )
    monkeypatch.setattr(stp_main, "_train_model", train_model)


async def _drain_pending_callbacks() -> None:
    """Let the event loop run done callbacks scheduled by finished tasks."""
    for _ in range(10):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_done_task_removed(monkeypatch, tmp_path):
    """A finished training task must be removed from model_train_tasks.

    The done callback used partial() with a lambda whose parameters were
    bound in the wrong order, so it popped the Task object instead of the
    model id. The finished task then stayed in model_train_tasks forever
    and every later retrain pass was silently skipped as "already
    training".
    """
    calls: List[Any] = []

    async def fake_train_model(
        model: Any, settings: Any, hass_info: Any, force_retrain: bool = False
    ) -> None:
        calls.append(model.id)

    _patch_retrain(monkeypatch, fake_train_model)
    state = _make_state(tmp_path)

    await stp_main._retrain_once(state)
    task = state.model_train_tasks[FAKE_MODEL_ID]
    await task
    await _drain_pending_callbacks()

    assert (
        FAKE_MODEL_ID not in state.model_train_tasks
    ), "finished training task was not removed from model_train_tasks"

    # A later retrain must not be skipped as "already training"
    await stp_main._retrain_once(state)
    await _drain_pending_callbacks()

    assert len(calls) == 2, (
        f"expected 2 trainings, got {len(calls)}; "
        "the finished task blocked later retrains"
    )


@pytest.mark.asyncio
async def test_failed_task_removed(monkeypatch, tmp_path):
    """A failed training task must also be removed from model_train_tasks.

    A training that raises (HA down mid-training, bad custom sentences)
    used to leave its task behind, so retraining never ran again until the
    server was restarted.
    """
    calls: List[Any] = []

    async def failing_train_model(
        model: Any, settings: Any, hass_info: Any, force_retrain: bool = False
    ) -> None:
        calls.append(model.id)
        raise RuntimeError("training failed")

    _patch_retrain(monkeypatch, failing_train_model)
    state = _make_state(tmp_path)

    await stp_main._retrain_once(state)
    task = state.model_train_tasks[FAKE_MODEL_ID]
    with pytest.raises(RuntimeError):
        await task
    await _drain_pending_callbacks()

    assert (
        FAKE_MODEL_ID not in state.model_train_tasks
    ), "failed training task was not removed from model_train_tasks"

    await stp_main._retrain_once(state)
    await _drain_pending_callbacks()

    assert len(calls) == 2, (
        f"expected 2 training attempts, got {len(calls)}; "
        "the failed task blocked later retrains"
    )
