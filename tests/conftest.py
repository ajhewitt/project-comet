import os

import pytest


def pytest_collection_modifyitems(config, items):
    # In CI mode, skip anything marked heavy
    if os.getenv("COMET_TEST_MODE") == "CI":
        skip_heavy = pytest.mark.skip(reason="Skipping heavy tests in CI mode")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
