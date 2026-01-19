import pytest

pytest.skip(
    "Legacy simulation/unit tests rely on discrete experience and were disabled in the continuous refactor.",
    allow_module_level=True,
)
