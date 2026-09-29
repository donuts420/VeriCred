import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image


def test_no_weights_returns_unavailable(tmp_path):
    from detection.predict import predict
    img = Image.new("RGB", (64, 64), "white")
    res = predict(img, conf=0.5, weights_path=str(tmp_path / "missing.pt"))
    assert res["model_loaded"] is False
    assert res["boxes"] == []
    assert res["fake_count"] == 0 and res["true_count"] == 0
    assert res["max_fake_conf"] == 0.0
    assert res["conf"] == 0.5


def test_result_schema_keys_without_weights():
    from detection.predict import predict
    res = predict(Image.new("RGB", (32, 32), "white"),
                  weights_path="/nonexistent/best.pt")
    assert set(res) >= {"boxes", "fake_count", "true_count",
                        "max_fake_conf", "annotated", "model_loaded", "conf"}


def test_class_names_match_dataset():
    from detection.predict import CLASS_NAMES
    assert CLASS_NAMES == {0: "fake", 1: "true"}
