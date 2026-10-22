# Развёртывание: сервер 2 × RTX 5090, 25 камер

Конфиги к `docs/SCALING_TZ.md` (§5 — что и почему, §6 — этапы и приёмка).
Порядок ниже — для чистого Ubuntu 24.04.

```
systemd/aimekeme@.service    процесс AI-API, шаблон: aimekeme@ai-g0 / @ai-g1a / @ai-g1b
systemd/ai-g0.env ...        карта и порт каждого процесса -> /etc/aimekeme/
systemd/gpu-clocks.*         persistence mode + фиксированные частоты GPU
systemd/vision-backend.service, go2rtc.service — платформа
nginx/aimekeme.conf          AI-API на :8000, раскладка по процессам и резерв
```

Допущения (поменять в файлах, если на сервере иначе): репозитории в
`/opt/aimekeme` и `/opt/vision`, всё работает от системного пользователя
`vision` с домашним каталогом `/var/lib/vision` (там кэш весов моделей).

## 1. Система

```bash
sudo useradd --system --create-home --home-dir /var/lib/vision vision
nvidia-smi                       # драйвер >= 570, видны обе 5090
ffmpeg -version | head -1        # нужен ffmpeg >= 5
ffmpeg -hide_banner -hwaccels | grep cuda
ffmpeg -hide_banner -filters | grep scale_cuda
```
Если `cuda`/`scale_cuda` нет — поставить статическую сборку ffmpeg с CUDA и
положить её в `PATH` раньше системной. ffmpeg 4.x не годится: там ключ
`-timeout` у RTSP значит другое (`vision/.../ffmpeg_source.py`).

## 2. AI-API

```bash
cd /opt/aimekeme
uv venv .venv && uv pip install --python .venv -r req.txt
uv pip install --python .venv onnxruntime-gpu==1.23.2
uv pip install --python .venv --no-deps insightface==1.0.1   # почему так — README.md
cp .env.example .env             # общие настройки моделей; DEVICE задаётся в /etc/aimekeme/*.env
sudo chown -R vision:vision /opt/aimekeme/known_faces
```
Проверка, что face поднимается на CUDA (иначе процесс не стартует — так и надо):
```bash
sudo -u vision env HOME=/var/lib/vision CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  .venv/bin/python -m uvicorn app:app --port 8199     # ждать "Application startup complete", Ctrl+C
```

## 3. Платформа

По `vision/DEPLOYMENT.md`, в `platform/backend/.env`:
```
AI_API_URL=http://127.0.0.1:8000
DECODE_HWACCEL=cuda
DECODE_GPU_COUNT=2
```

## 4. Юниты

```bash
sudo mkdir -p /etc/aimekeme
sudo cp /opt/aimekeme/deploy/systemd/*.env /etc/aimekeme/
sudo cp /opt/aimekeme/deploy/systemd/*.service /etc/systemd/system/
nvidia-smi -q -d SUPPORTED_CLOCKS | grep -m5 Graphics   # выбрать диапазон
sudo nano /etc/aimekeme/gpu-clocks.env                    # заполнить GPU_CLOCKS
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-clocks \
    aimekeme@ai-g0 aimekeme@ai-g1a aimekeme@ai-g1b go2rtc vision-backend
```

## 5. nginx

```bash
sudo apt install -y nginx
sudo cp /opt/aimekeme/deploy/nginx/aimekeme.conf /etc/nginx/conf.d/
sudo nginx -t && sudo systemctl reload nginx
```

## 6. Проверка

```bash
systemctl --failed                                   # пусто
for p in 8100 8101 8102; do curl -s :$p/health; echo; done   # у всех "ok", device "cuda"
curl -s :8000/health                                 # то же через nginx
nvidia-smi                                           # 3 процесса python: 1 на GPU0, 2 на GPU1
journalctl -u aimekeme@ai-g0 -n 30                   # "Device: cuda (NVIDIA GeForce RTX 5090)", warm-up
cd /opt/vision/platform/backend && venv/bin/python tests/test_load.py \
    --ai-stats http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102
```
Отчёт `test_load.py` сам предупреждает о том, что выходит за цели ТЗ:
пропуски тактов > 1%, возраст кадра выше цели, очередь к модели > 20%,
загрузка GPU > 60%, декодера > 70%, троттлинг.

## Сколько процессов

Раскладка выше — стартовая. Сколько процессов нужно на самом деле, решает
замер этапа 0: `scripts/bench_models.py <кадр аудитории> --procs 2` на каждой
карте (`CUDA_VISIBLE_DEVICES=0`, `=1`). Добавить процесс — новый файл
`/etc/aimekeme/ai-gXy.env` (свои карта и порт), `systemctl enable --now
aimekeme@ai-gXy` и строка `server 127.0.0.1:<порт> ...` в нужном upstream
nginx. Убрать — наоборот.
