#!/usr/bin/env bash
# Sample the run-benchmark process host RSS + GPU memory every 5s → a log we can read remotely.
OUT="${1:-mem_sample.log}"
: > "$OUT"
while sleep 5; do
  pid=$(pgrep -f "viscurate.cli run-benchmark" | head -1)
  rss="NA"
  [ -n "$pid" ] && rss=$(awk '/VmRSS/{printf "%.1f", $2/1024/1024}' "/proc/$pid/status" 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  printf '%s pid=%s host_rss_gb=%s gpu_mb=%s\n' "$(date +%T)" "${pid:-none}" "${rss:-NA}" "${gpu:-NA}" >> "$OUT"
done
