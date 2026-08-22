#!/usr/bin/env bash
set -euo pipefail

MINER="${AKOYA_MINER_BIN:-./akoya-miner}"
GPU="${AKOYA_GPU_INDICES:-2}"
M="${AKOYA_MINE_M:-4096}"
N="${AKOYA_MINE_N:-131072}"
K="${AKOYA_MINE_K:-4096}"
R="${AKOYA_MINE_NOISE_RANK:-128}"
REPEATS="${1:-5}"
DURATION="${2:-4}"
PROFILE_CALL="${3:-100}"
OUT="${AKOYA_BREAKDOWN_OUT:-akoya-breakdown-$(date +%Y%m%d-%H%M%S).log}"

if [[ ! -x "$MINER" ]]; then
  echo "ERROR: miner not executable: $MINER" >&2
  exit 1
fi

: > "$OUT"

echo "Akoya native stage breakdown" | tee -a "$OUT"
echo "GPU=$GPU M=$M N=$N K=$K R=$R repeats=$REPEATS duration=${DURATION}s profile_call=$PROFILE_CALL" | tee -a "$OUT"
echo "NOTE: overall PERF-ITER throughput from these runs is intentionally perturbed by the one-shot cudaEvent synchronization; use only the PEARL NATIVE PROFILE timings." | tee -a "$OUT"
echo | tee -a "$OUT"

for ((i=1; i<=REPEATS; i++)); do
  echo "===== sample $i/$REPEATS =====" | tee -a "$OUT"

  tmp="$(mktemp)"
  set +e
  PEARL_GEMM_PROFILE_ITER="$PROFILE_CALL" \
  AKOYA_GPU_INDICES="$GPU" \
  AKOYA_MINE_M="$M" \
  AKOYA_MINE_N="$N" \
  AKOYA_MINE_K="$K" \
  AKOYA_MINE_NOISE_RANK="$R" \
  AKOYA_DISABLE_PONG=1 \
  "$MINER" perf-iter "$DURATION" 1 >"$tmp" 2>&1
  rc=$?
  set -e

  grep -E 'PEARL NATIVE PROFILE (NOISY|ITER)|PERF-ITER RESULT' "$tmp" | tee -a "$OUT" || true

  if ! grep -q 'PEARL NATIVE PROFILE ITER' "$tmp"; then
    echo "ERROR: native profile line missing in sample $i (rc=$rc)." | tee -a "$OUT" >&2
    echo "Possible causes: old libpearl_gemm_capi.so, PROFILE_CALL not reached, or wrong native library loaded." | tee -a "$OUT" >&2
    echo "Full sample log follows:" >> "$OUT"
    cat "$tmp" >> "$OUT"
    rm -f "$tmp"
    exit 2
  fi

  rm -f "$tmp"
  echo | tee -a "$OUT"
done

awk '
function grab(line, key,    x) {
  x = line
  if (index(x, key "=") == 0) return -1
  sub(".*" key "=", "", x)
  sub("ms.*", "", x)
  return x + 0
}
/PEARL NATIVE PROFILE NOISY:/ {
  ax = grab($0, "AxEBL")
  ap = grab($0, "ApEA")
  tg = grab($0, "transcript_gemm")
  nt = grab($0, "total")
  if (ax >= 0 && ap >= 0 && tg >= 0 && nt >= 0) {
    sax += ax; sap += ap; stg += tg; snt += nt; nn++
  }
}
/PEARL NATIVE PROFILE ITER/ {
  lcg = grab($0, "LCG")
  th  = grab($0, "TensorHash")
  ch  = grab($0, "CommitHash")
  ng  = grab($0, "NoiseGenA")
  gem = grab($0, "NoisyGemm")
  tot = grab($0, "Total")
  if (lcg >= 0 && th >= 0 && ch >= 0 && ng >= 0 && gem >= 0 && tot >= 0) {
    slcg += lcg; sth += th; sch += ch; sng += ng; sgem += gem; stot += tot; ni++
  }
}
END {
  print "===== AVERAGE NATIVE STAGE BREAKDOWN ====="
  if (ni > 0) {
    printf "LCG             %8.3f ms  %6.2f%%\n", slcg/ni, 100*(slcg/ni)/(stot/ni)
    printf "TensorHash      %8.3f ms  %6.2f%%\n", sth/ni,  100*(sth/ni)/(stot/ni)
    printf "CommitHash      %8.3f ms  %6.2f%%\n", sch/ni,  100*(sch/ni)/(stot/ni)
    printf "NoiseGenA       %8.3f ms  %6.2f%%\n", sng/ni,  100*(sng/ni)/(stot/ni)
    printf "NoisyGemm       %8.3f ms  %6.2f%%\n", sgem/ni, 100*(sgem/ni)/(stot/ni)
    printf "TOTAL           %8.3f ms\n", stot/ni
  } else {
    print "No ITER samples parsed."
  }
  print ""
  if (nn > 0) {
    print "----- NoisyGemm detail -----"
    printf "AxEBL           %8.3f ms  %6.2f%% of NoisyGemm\n", sax/nn, 100*(sax/nn)/(snt/nn)
    printf "ApEA            %8.3f ms  %6.2f%% of NoisyGemm\n", sap/nn, 100*(sap/nn)/(snt/nn)
    printf "transcript_gemm %8.3f ms  %6.2f%% of NoisyGemm\n", stg/nn, 100*(stg/nn)/(snt/nn)
    printf "NOISY TOTAL     %8.3f ms\n", snt/nn
  } else {
    print "No NOISY samples parsed."
  }
}
' "$OUT" | tee -a "$OUT"

echo
echo "Saved: $OUT"
