#!/bin/bash
# 24-часовой прогон под полной нагрузкой (docs/SCALING_TZ.md, R1, этап 2).
# Раз в INTERVAL секунд:
#   - полный отчёт vision tests/test_load.py -> $OUT/<время>.txt
#   - строка в $OUT/soak.log: память и число перезапусков каждого сервиса,
#     память/загрузка/температура/троттлинг каждой карты, пропуски тактов
# Запуск:  nohup bash /opt/aimekeme/deploy/soak.sh > /dev/null 2>&1 &
# Итог:    bash /opt/aimekeme/deploy/soak.sh --summary
set -u
OUT="${SOAK_DIR:-$HOME/soak}"
INTERVAL="${SOAK_INTERVAL:-1800}"
SERVICES="aimekeme@ai-g0 aimekeme@ai-g0b aimekeme@ai-g1a aimekeme@ai-g1b vision-backend go2rtc"
AI_STATS="http://127.0.0.1:8100,http://127.0.0.1:8103,http://127.0.0.1:8101,http://127.0.0.1:8102"

if [ "${1:-}" = "--summary" ]; then
    echo "== первая и последняя строки (рост памяти = утечка, R1: <= 5% после первого часа)"
    head -1 "$OUT/soak.log"; tail -1 "$OUT/soak.log"
    echo "== предупреждения во всех отчётах (сколько раз)"
    grep -h "⚠" "$OUT"/2*.txt | sed 's/[0-9.]\+ мс/N мс/g' | sort | uniq -c | sort -rn | head -20
    exit 0
fi

mkdir -p "$OUT"
while true; do
    ts=$(date +%F_%H%M)
    (cd /opt/vision/platform/backend && venv/bin/python tests/test_load.py \
        --out-dir "$OUT" --ai-stats "$AI_STATS" > /dev/null 2>&1)
    report=$(ls -t "$OUT"/2*.txt 2>/dev/null | head -1)
    skips=$(grep -oE 'пропущено +[0-9]+' "$report" 2>/dev/null | awk '{s+=$2} END {print s+0}')
    line="$ts skips=$skips"
    for s in $SERVICES; do
        mem=$(systemctl show -p MemoryCurrent --value "$s")
        restarts=$(systemctl show -p NRestarts --value "$s")
        # anon — память программ (утечка видна здесь), file — кэш файлов (не утечка)
        stat="/sys/fs/cgroup$(systemctl show -p ControlGroup --value "$s")/memory.stat"
        anon=$(awk '$1=="anon" {print int($2/1048576)}' "$stat" 2>/dev/null)
        file=$(awk '$1=="file" {print int($2/1048576)}' "$stat" 2>/dev/null)
        line="$line | $s mem=$(( ${mem:-0} / 1048576 ))MB anon=${anon}MB file=${file}MB restarts=$restarts"
    done
    # внутри vision-backend: сам python и 23 ffmpeg по отдельности
    py=$(ps -o rss= -p "$(systemctl show -p MainPID --value vision-backend)" 2>/dev/null)
    ff=$(ps -o rss= -C ffmpeg 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
    line="$line | backend-python=$(( ${py:-0} / 1024 ))MB ffmpeg-total=${ff:-0}MB"
    gpus=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active \
           --format=csv,noheader,nounits | tr '\n' ';')
    echo "$line | gpu(idx,memMB,util%,°C,throttle) $gpus" >> "$OUT/soak.log"
    sleep "$INTERVAL"
done
