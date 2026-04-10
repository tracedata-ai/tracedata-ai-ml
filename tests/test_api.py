import json
import random
import socket
from datetime import datetime, timedelta

import pytest
import requests


def _api_listening(host: str = "localhost", port: int = 8000, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.integration
def test_api():
    """Requires a running HTTP API on localhost:8000 (e.g. your FastAPI app)."""
    if not _api_listening():
        pytest.skip("No server on localhost:8000 (start the API to run this test)")

    url = "http://localhost:8000/score-trip"

    telemetry = []
    for i in range(10):
        telemetry.append(
            {
                "timestamp": (datetime.now() + timedelta(seconds=i * 30)).isoformat(),
                "speed_kmh": 40 + random.uniform(-5, 5),
                "acceleration_ms2": random.uniform(-0.3, 0.3),
                "lat": 1.35,
                "lon": 103.8,
            }
        )

    payload = {
        "driver_id": 1,
        "truck_id": 42,
        "telemetry": telemetry,
    }

    response = requests.post(url, json=payload, timeout=10)
    assert response.status_code == 200, response.text

    stats_response = requests.get("http://localhost:8000/driver/1", timeout=10)
    assert stats_response.status_code == 200, stats_response.text
    # Valid JSON body
    json.loads(stats_response.text)


if __name__ == "__main__":
    test_api()
