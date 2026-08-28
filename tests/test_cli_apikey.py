from __future__ import annotations

from typer.testing import CliRunner

import n2n.cli as cli_module
from n2n.auth import ApiKeyStore

runner = CliRunner()


def _isolated_store(tmp_path, monkeypatch):
    test_store = ApiKeyStore(path=tmp_path / "api_keys.json")
    monkeypatch.setattr(cli_module, "api_key_store", test_store)
    return test_store


def test_apikey_list_empty(tmp_path, monkeypatch):
    _isolated_store(tmp_path, monkeypatch)
    result = runner.invoke(cli_module.app, ["apikey", "list"])
    assert result.exit_code == 0
    assert "No API keys" in result.output


def test_apikey_create_prints_plaintext_once(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    result = runner.invoke(cli_module.app, ["apikey", "create", "--name", "ci-bot"])
    assert result.exit_code == 0
    assert "n2n_live_" in result.output
    assert "ci-bot" in result.output
    assert len(store.list()) == 1


def test_apikey_list_shows_created_key_without_plaintext_or_hash(tmp_path, monkeypatch):
    _isolated_store(tmp_path, monkeypatch)
    runner.invoke(cli_module.app, ["apikey", "create", "--name", "ci-bot"])
    result = runner.invoke(cli_module.app, ["apikey", "list"])
    assert "ci-bot" in result.output
    assert "active" in result.output
    assert "n2n_live_" not in result.output  # never re-displayed


def test_apikey_revoke_known_id(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    runner.invoke(cli_module.app, ["apikey", "create", "--name", "ci-bot"])
    key_id = store.list()[0].id
    result = runner.invoke(cli_module.app, ["apikey", "revoke", key_id])
    assert result.exit_code == 0
    assert store.list()[0].revoked is True


def test_apikey_revoke_unknown_id_exits_nonzero(tmp_path, monkeypatch):
    _isolated_store(tmp_path, monkeypatch)
    result = runner.invoke(cli_module.app, ["apikey", "revoke", "not-a-real-id"])
    assert result.exit_code != 0
