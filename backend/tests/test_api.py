def test_health_without_ollama(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["pack"] == "church"
    assert body["ollama"]["available"] is False


def test_workflows_listed(client):
    ids = {w["id"] for w in client.get("/api/workflows").json()}
    assert "sermon_studio" in ids


def test_upload_ask_and_audit(client):
    r = client.post(
        "/api/library",
        files={"file": ("policy.md", b"The fellowship hall seats 120 people.", "text/markdown")},
    )
    assert r.status_code == 200, r.text
    assert client.get("/api/library").json()[0]["title"] == "policy"

    r = client.post("/api/ask", json={"question": "How many seats in the fellowship hall?"})
    assert r.status_code == 200, r.text
    assert r.json()["citations"][0]["title"] == "policy"

    actions = [e["action"] for e in client.get("/api/audit").json()]
    assert "ask" in actions and "library.upload" in actions


def test_rejects_unknown_file_type(client):
    r = client.post("/api/library", files={"file": ("x.exe", b"nope", "application/octet-stream")})
    assert r.status_code == 400


def test_root_without_frontend_build(client):
    assert "backend is running" in client.get("/").json()["message"]
