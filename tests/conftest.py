import os

import pytest

from taxverity.config import ENV_PREFIX


# Session-scoped, and autouse so it is ordered ahead of every other fixture in
# this scope: the session-scoped corpus extraction resolves its path during
# setup, which is too early for a function-scoped patch to have cleaned up.
@pytest.fixture(scope="session", autouse=True)
def isolated_environment():
    with pytest.MonkeyPatch.context() as patcher:
        for name in [key for key in os.environ if key.startswith(ENV_PREFIX)]:
            patcher.delenv(name, raising=False)
        yield
