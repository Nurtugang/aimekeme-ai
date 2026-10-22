"""FastAPI-приложение: модульный монолит системы видеонаблюдения.

Один процесс, один общий torch. Все модели грузятся ОДИН раз при старте
(lifespan) и складываются в app.state.detectors; роутеры берут готовые
детекторы оттуда через request.app.state — без загрузки на запрос.
"""

import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import stats
from app.counting.detector import CountingDetector
from app.counting.router import router as counting_router
from app.face.detector import FaceDetector
from app.face.router import router as face_router
from app.fight.detector import FightDetector
from app.fight.router import router as fight_router
from app.fire.detector import FireDetector
from app.fire.router import router as fire_router
from app.persons.detector import PersonsDetector
from app.persons.router import router as persons_router
from app.config import settings

# Глобальная настройка логирования.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# huggingface_hub на каждой загрузке весов сыплет HTTP-запросами в INFO — это вода.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("surveillance")


def resolve_device(preference: str) -> torch.device:
    """auto -> cuda при наличии, иначе cpu; либо явное значение (cuda/cpu/cuda:0)."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device(settings.device)
    # На сервере несколько процессов, у каждого своя карта (CUDA_VISIBLE_DEVICES):
    # по этой строке видно, какая именно.
    logger.info("Device: %s (%s)", device,
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")

    fight = FightDetector(settings, device)
    fight.load()

    face = FaceDetector(settings)
    face.load()

    fire = FireDetector(settings, device)
    fire.load()

    counting = CountingDetector(settings, device)
    counting.load()

    persons = PersonsDetector(settings, device)
    persons.load()

    # Кладём готовые модели на app.state — отсюда их берут роутеры.
    app.state.detectors = {
        "fight": fight,
        "face": face,
        "fire": fire,
        "counting": counting,
        "persons": persons,
    }

    yield


app = FastAPI(
    title="Intelligent Surveillance API",
    description="Алгоритмы видеонаблюдения: детекция драк, распознавание лиц, "
    "детекция огня/дыма, подсчёт людей и детекция людей с цветом одежды.",
    version=settings.api_version,
    lifespan=lifespan,
)

@app.middleware("http")
async def time_detect_requests(request: Request, call_next):
    """Полное время запроса /detect/* — с разбором JSON и ожиданием свободного
    потока, для GET /stats. Остальные пути не меряем: у /faces/{id} ключей было
    бы столько же, сколько людей."""
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/detect/"):
        stats.record(request.url.path, "total_ms", (time.perf_counter() - start) * 1000.0)
    return response


app.include_router(fight_router)
app.include_router(face_router)
app.include_router(fire_router)
app.include_router(counting_router)
app.include_router(persons_router)


class ModelHealth(BaseModel):
    name: str
    ready: bool
    device: str


class HealthResponse(BaseModel):
    status: str
    models: list[ModelHealth]
    version: str


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Готовность: какие модели загружены и на каком устройстве."""
    models = [
        ModelHealth(name=name, ready=det.is_ready, device=det.device)
        for name, det in app.state.detectors.items()
    ]
    all_ready = all(m.ready for m in models) and bool(models)
    return HealthResponse(
        status="ok" if all_ready else "loading",
        models=models,
        version=settings.api_version,
    )


class Percentiles(BaseModel):
    count: int
    p50: float
    p95: float


@app.get("/stats", response_model=dict[str, dict[str, Percentiles]], tags=["system"])
def get_stats() -> dict[str, dict[str, Percentiles]]:
    """Замеры с рестарта процесса, мс (docs/SCALING_TZ.md §4.4):

    - `/detect/<name>` -> `total_ms` — полное время запроса;
    - `<модель>` -> `lock_wait_ms` (ждал lock — очередь), `lock_held_ms`
      (держал lock — сама модель).

    count — всего замеров с рестарта; p50/p95 — по последним 2048.
    """
    return stats.snapshot()
