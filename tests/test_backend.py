import time

from fastapi.testclient import TestClient

from backend.app import Plant, create_app


def test_web_app(demo_data, monkeypatch):
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(demo_data))
    plant = Plant(history_days=2, speed=600)
    client = TestClient(create_app(plant))

    assert client.get("/").status_code == 200
    assert client.get("/api/config").json()["speed"] == 600
    summary = client.get("/api/summary").json()
    assert summary["parts"] > 10_000 and 0 < summary["defects"] < summary["parts"]

    machines = client.get("/api/machines").json()["machines"]
    assert [m["id"] for m in machines] == [f"IMM-0{i}" for i in range(1, 7)]
    assert {m["status"] for m in machines} <= {"ok", "watch", "alarm", "stopped"}

    deadline = time.monotonic() + 10  # live production keeps going after the history
    while not plant.live and time.monotonic() < deadline:
        time.sleep(0.1)
    assert plant.live and plant.live[0]["timestamp"] >= plant.df.timestamp.iloc[0]

    rows = client.get("/api/search", params={"result": "defect", "limit": 5}).json()["rows"]
    assert len(rows) == 5 and all(r["true_class"] for r in rows)
    part = client.get(f"/api/parts/{rows[0]['part_id']}").json()
    assert part["cls"] == rows[0]["true_class"] and len(part["views"]) == 5
    assert client.get(part["views"][0]["url"]).headers["content-type"] == "image/jpeg"

    assert client.get("/api/samples/999999/0").status_code == 404
    assert client.get("/api/parts/ZD-NOPE").status_code == 404
    assert client.get("/api/machines/IMM-99/history").status_code == 404
