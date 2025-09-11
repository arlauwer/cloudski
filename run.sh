#!/bin/bash

MAX_JOBS=32
ulimit -n 20000
CLOUDY_BIN="cloudy"
FORCE=false

# parse options
while getopts "f" opt; do
  case $opt in
    f) FORCE=true ;;
  esac
done

cd runs || exit 1

run_sim() {
  local dir="$1"
  cd "$dir" || return
  echo "Starting $dir"
  $CLOUDY_BIN < sim.in > sim.out
  echo "Finished $dir"
}

jobcount=0
for dir in */; do
  if [ -f "$dir/sim.out" ] && [ "$FORCE" = false ]; then
    echo "Skipping $dir (already run)"
    continue
  fi

  run_sim "$dir" &
  ((jobcount++))

  if (( jobcount >= MAX_JOBS )); then
    wait -n
    ((jobcount--))
  fi
done

wait
echo "All simulations complete."

