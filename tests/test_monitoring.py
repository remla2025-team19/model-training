import pytest, time

MAX_MS = 50

@pytest.mark.monitoring
def test_inference_latency(model):
    txt = "Really loved the ambience and the dessert!"
    start = time.perf_counter()
    _ = model(txt)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    assert elapsed_ms < MAX_MS, f"Inference took {elapsed_ms:.1f} ms > {MAX_MS}"
