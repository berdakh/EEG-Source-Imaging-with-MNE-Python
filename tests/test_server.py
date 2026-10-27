import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EEGAGENT_PROVIDER", "rule")
    from eegagent import server

    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    server._agents.clear()
    server._locks.clear()
    return TestClient(server.app)


def test_health_and_tools(client):
    assert client.get("/health").json()["status"] == "ok"
    names = {t["name"] for t in client.get("/tools").json()}
    assert "compute_sources" in names


def test_session_chat_state_report(client):
    sid = client.post("/sessions").json()["session_id"]
    r = client.post(
        f"/sessions/{sid}/chat",
        json={"message": "Load the simulated data, localise the sources and make a report"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["name"] for e in body["trace"]][:2] == ["load_recording", "make_epochs"]
    assert body["state"]["sources"]["peak"]["position_mm"][0] < 0
    assert client.get(f"/sessions/{sid}/state").json()["epochs"]["n_epochs"] > 0
    html = client.get(f"/sessions/{sid}/report")
    assert html.status_code == 200 and "<html" in html.text.lower()
    # follow-up turn reuses the session
    r2 = client.post(f"/sessions/{sid}/chat", json={"message": "run a cluster test"})
    assert "cluster_test" in [e["name"] for e in r2.json()["trace"]]


def test_unknown_session_and_validation(client):
    assert client.get("/sessions/nope/state").status_code == 404
    sid = client.post("/sessions").json()["session_id"]
    assert client.get(f"/sessions/{sid}/report").status_code == 404
    assert client.post(f"/sessions/{sid}/chat", json={"message": ""}).status_code == 422
    assert client.delete(f"/sessions/{sid}").json()["deleted"] == sid
