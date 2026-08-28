"""Local Ed25519 keypair for manifest signing.

No hosted signing service — the customer owns their own key, generated on
first run and stored locally. This is deliberate: a hosted signer would
reintroduce the cloud dependency the product exists to avoid.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

DEFAULT_KEY_DIR = Path.home() / ".n2n" / "keys"
PRIVATE_KEY_FILE = "signing_key.pem"
PUBLIC_KEY_FILE = "signing_key.pub"


def load_or_create_keypair(key_dir: Path = DEFAULT_KEY_DIR) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    key_dir.mkdir(parents=True, exist_ok=True)
    private_path = key_dir / PRIVATE_KEY_FILE
    public_path = key_dir / PUBLIC_KEY_FILE

    if private_path.exists():
        private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
        return private_key, private_key.public_key()

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(private_bytes)
    private_path.chmod(0o600)
    public_path.write_bytes(public_bytes)
    return private_key, private_key.public_key()


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    import hashlib

    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()
