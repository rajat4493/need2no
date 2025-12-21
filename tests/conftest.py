import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def pytest_sessionstart(session):
    import n2n  # noqa: WPS433 (import inside function for guard)

    resolved = str(pathlib.Path(n2n.__file__).resolve())
    if ".venv/src/n2n-redactor" in resolved:
        raise RuntimeError(
            f"Pytest is importing stale n2n from {resolved}. "
            "Run: pip uninstall -y n2n-redactor && pip install -e ."
        )

