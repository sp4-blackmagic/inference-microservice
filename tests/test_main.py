# Import the app from your src package
# This works because src is now a package
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_hello_world():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_evaluate():
    info_data = {
        "file_uid": "1",
        "models": ["lstm"]
    }
    response = client.post("/evaluate/", json=info_data)

    assert response.status_code == 200
