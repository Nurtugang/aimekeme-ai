"""Векторы лиц в ответах AI-API: /detect/face и /embed.

Запуск: python tests/test_face_embed.py   (нужна GPU или DEVICE=cpu; модель buffalo_l
качается в ~/.insightface при первом запуске). Фото — примеры, которые идут с insightface.
"""

import base64
import math
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import insightface  # noqa: E402
from PIL import Image  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings, settings  # noqa: E402
from app.face.detector import FaceDetector  # noqa: E402
from app.face.router import router  # noqa: E402

_IMAGES = Path(insightface.__file__).parent / "data" / "images"

detector = FaceDetector(Settings())
detector.load()
_app = FastAPI()
_app.include_router(router)
_app.state.detectors = {"face": detector}
client = TestClient(_app)


def _png(name: str) -> bytes:
    return (_IMAGES / name).read_bytes()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _crop_one_face() -> bytes:
    """Самое крупное лицо из группового t1.jpg с запасом вокруг, PNG с альфа-каналом."""
    image = Image.open(_IMAGES / "t1.jpg").convert("RGB")
    faces = client.post("/detect/face", json={"frame": _b64(_png("t1.jpg"))}).json()["faces"]
    x1, y1, x2, y2 = max((f["box"] for f in faces), key=lambda b: b[2] - b[0])
    pad = x2 - x1
    crop = image.crop((int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad)))
    out = BytesIO()
    crop.convert("RGBA").save(out, "PNG")
    return out.getvalue()


ONE_FACE = _crop_one_face()


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_detect_face_has_embedding():
    r = client.post("/detect/face", json={"frame": _b64(ONE_FACE)})
    assert r.status_code == 200, r.text
    face = r.json()["faces"][0]
    assert len(face["embedding"]) == 512
    assert abs(_norm(face["embedding"]) - 1.0) < 1e-3
    assert "identity" not in face                 # кто это — решает vision


def test_embedding_rounded_to_6_digits():
    r = client.post("/detect/face", json={"frame": _b64(ONE_FACE)})
    emb = r.json()["faces"][0]["embedding"]
    assert all(round(x, 6) == x for x in emb), emb[:3]   # короче JSON, косинус не страдает


def test_detect_face_batch_has_embedding():
    b = _b64(ONE_FACE)
    r = client.post("/detect/face", json={"frames": [b, b]})
    assert r.status_code == 200, r.text
    assert all(len(res["faces"][0]["embedding"]) == 512 for res in r.json()["results"])


def _blank_jpeg() -> bytes:
    out = BytesIO()
    Image.new("RGB", (400, 400), "white").save(out, "JPEG")
    return out.getvalue()


def test_embed_one_face():
    r = client.post("/embed", files={"image": ("f.png", ONE_FACE)})
    assert r.status_code == 200, r.text
    emb = r.json()["embedding"]
    assert len(emb) == 512 and abs(_norm(emb) - 1.0) < 1e-3
    det = client.post("/detect/face", json={"frame": _b64(ONE_FACE)})
    same = sum(a * b for a, b in zip(emb, det.json()["faces"][0]["embedding"]))
    assert same > 0.9, same                       # тот же человек тем же путём


def test_embed_many_faces():
    r = client.post("/embed", files={"image": ("t1.jpg", _png("t1.jpg"))})
    assert r.status_code == 422 and r.json()["detail"] == "more than one face", r.text


def test_embed_no_face():
    r = client.post("/embed", files={"image": ("blank.jpg", _blank_jpeg())})
    assert r.status_code == 422 and r.json()["detail"] == "no face", r.text


def test_embed_bad_file():
    r = client.post("/embed", files={"image": ("x.jpg", b"not an image")})
    assert r.status_code == 422 and r.json()["detail"] == "invalid image", r.text


def test_embed_too_large():
    big = b"0" * (settings.face_max_upload_bytes + 1)
    r = client.post("/embed", files={"image": ("big.jpg", big)})
    assert r.status_code == 413, r.text


def test_faces_db_endpoints_removed():
    assert client.get("/faces").status_code == 404
    assert client.post("/faces", data={"name": "x"}).status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
