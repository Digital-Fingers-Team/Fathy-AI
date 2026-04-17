from fastapi.testclient import TestClient

from app.main import create_app


def main() -> None:
    app = create_app()
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200, r.text

    r = client.post("/teach", json={"question": "What is Fathy?", "answer": "A memory-first assistant.", "tags": ["intro"]})
    assert r.status_code == 200, r.text
    mem_id = r.json()["id"]

    r = client.get("/memory")
    assert r.status_code == 200, r.text
    assert r.json()["total"] >= 1

    r = client.post("/chat", json={"message": "What is Fathy?"})
    assert r.status_code == 200, r.text
    assert "Fathy" in r.json()["answer"] or "assistant" in r.json()["answer"]

    r = client.delete(f"/memory/{mem_id}")
    assert r.status_code == 200, r.text

    print("Smoke test OK")


if __name__ == "__main__":
    main()

