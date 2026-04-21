import os
import sys
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[2] / "server"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

os.chdir(SERVER_DIR)


@pytest.fixture
def app():
    from server_app.app_factory import create_app

    app = create_app()
    app.testing = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()
