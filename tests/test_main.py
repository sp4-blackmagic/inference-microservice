# Import the app from your src package
# This works because src is now a package
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_test():
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"msg": "Its working!"}


def _test_test_cluster():
    response = client.get("/test_cluster")

    assert response.status_code == 200
    assert response.json() == {"status": "Cluster is not reachable"}


def test_evaluate():
    info_data = {
        "file_uid": "1",
        "models": ["lstm"]
    }
    response = client.post("/evaluate/", json=info_data)

    assert response.status_code == 500
