import os
import sys
from pathlib import Path

import pytest


SERVER_DIR = Path('/Users/ryad/Code/delineo/Algorithms/server')

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

os.chdir(SERVER_DIR)

from server_app.app_factory import create_app  # noqa: E402


@pytest.fixture
def app():
    app = create_app()
    app.testing = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()
