"""Step 11.2 — password hashing (ADR-029, ADR-111).

argon2id through `argon2-cffi`, at the library's RFC 9106 low-memory profile
(t=3, m=64 MiB, p=4). The hash string carries its own parameters, so raising
them later needs no migration: `needs_rehash()` flags an old hash at login.
"""

from __future__ import annotations

from functools import cache

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# Long enough to resist guessing without a composition rule (NIST SP 800-63B
# recommends length over composition). The ceiling bounds the work one request
# can make the hasher do.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 1_024

_HASHER = PasswordHasher(type=Type.ID)


class WeakPassword(ValueError):
    """A password outside the length policy. Safe to show the user."""


def check_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPassword(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise WeakPassword(f"password must be at most {MAX_PASSWORD_LENGTH} characters")


def hash_password(password: str) -> str:
    check_policy(password)
    return _HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """False for a wrong password and for a malformed hash; never raises on either."""
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    try:
        return _HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _HASHER.check_needs_rehash(password_hash)


@cache
def dummy_hash() -> str:
    """A real hash to verify against when no account exists, so an unknown
    address costs the same time as a wrong password and timing reveals nothing."""
    return _HASHER.hash("no account has this password")
