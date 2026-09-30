import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app import Plant, create_app


def test_web_app(demo_data, monkeypatch):
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(demo_data))
    plant = Plant(history_days=1, interval=0.05)
    try:
        client = TestClient(create_app(plant))
        assert client.get("/").status_code == 200
        config = client.get("/api/config").json()
        assert config["part_types"] == ["end_cap", "mounts", "plastic_nut", "plastic_plug", "u_block"]
        model = client.get("/api/model").json()["part_types"]["plastic_nut"]
        assert model["parts"] > 0 and 0 <= model["accuracy"] <= 1

        deadline = time.monotonic() + 30  # the live line inspects parts with the model
        while not plant.events and time.monotonic() < deadline:
            time.sleep(0.1)
        event = plant.events[-1][1]
        assert set(event["result"]) >= {"defect", "cls", "score", "threshold", "ms"}
        assert len(event["views"]) == 5 and event["views"][0]["heatmap"].startswith("data:image/png")
        assert event["context"]["machine"].startswith("IMM-")  # placeholder production context

        img = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))[1].tobytes()
        res = client.post("/api/inspect", content=img).json()
        assert res["part_type"] in config["part_types"] and isinstance(res["defect"], bool)
        assert client.post("/api/inspect", content=b"not an image").status_code == 400
        assert client.post("/api/inspect?part_type=rocket", content=img).status_code == 400
        test_img = client.get("/api/test-image", params={"defect": True})
        assert test_img.headers["x-label"] != "good"

        assert client.get("/api/summary").json()["parts"] > 1000
        machines = client.get("/api/machines").json()["machines"]
        assert [m["id"] for m in machines] == [f"IMM-0{i}" for i in range(1, 7)]
        rows = client.get("/api/search", params={"result": "defect", "limit": 3}).json()["rows"]
        part = client.get(f"/api/parts/{rows[0]['part_id']}").json()  # history part: inspected on demand
        assert "result" in part and len(part["views"]) == 5
        assert client.get(part["views"][0]["url"]).headers["content-type"] == "image/jpeg"
        assert client.get("/api/samples/999999/0").status_code == 404
        assert client.get("/api/parts/ZD-NOPE").status_code == 404
    finally:
        plant.close()
