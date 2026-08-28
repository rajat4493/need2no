from __future__ import annotations

import json

from n2n.auth import ApiKeyStore


def test_create_returns_plaintext_once_and_never_persists_it(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    plaintext, record = store.create("ci-bot")

    assert plaintext.startswith("n2n_live_")
    assert record.name == "ci-bot"
    assert not record.revoked

    raw = json.loads(store.path.read_text())
    assert len(raw) == 1
    assert plaintext not in json.dumps(raw)  # plaintext never written to disk
    assert "hashed_key" in raw[0]
    assert raw[0]["hashed_key"] != plaintext


def test_keys_file_has_restrictive_permissions(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    store.create("ci-bot")
    mode = store.path.stat().st_mode & 0o777
    assert mode == 0o600


def test_verify_accepts_a_valid_key(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    plaintext, record = store.create("ci-bot")
    verified = store.verify(plaintext)
    assert verified is not None
    assert verified.id == record.id


def test_verify_rejects_garbage(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    store.create("ci-bot")
    assert store.verify("not-a-real-key") is None
    assert store.verify("") is None
    assert store.verify(None) is None  # type: ignore[arg-type]


def test_verify_rejects_a_revoked_key(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    plaintext, record = store.create("ci-bot")
    assert store.verify(plaintext) is not None

    assert store.revoke(record.id) is True
    assert store.verify(plaintext) is None


def test_revoke_unknown_id_returns_false(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    assert store.revoke("does-not-exist") is False


def test_verify_updates_last_used_at(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    plaintext, record = store.create("ci-bot")
    assert record.last_used_at is None
    store.verify(plaintext)
    assert store.list()[0].last_used_at is not None


def test_list_never_exposes_hash_via_public_dict(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    _, record = store.create("ci-bot")
    assert "hashed_key" not in record.public_dict()


def test_two_keys_have_distinct_plaintexts_and_ids(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    p1, r1 = store.create("a")
    p2, r2 = store.create("b")
    assert p1 != p2
    assert r1.id != r2.id


def test_is_empty(tmp_path):
    store = ApiKeyStore(path=tmp_path / "keys.json")
    assert store.is_empty() is True
    store.create("ci-bot")
    assert store.is_empty() is False
