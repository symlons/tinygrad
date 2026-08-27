#!/bin/bash
# Wrap a command with this to make tinygrad's NV backend select the GPU(s) actually
# reachable in this container, e.g.: modal_select_gpu.sh python3 my_gemm_script.py
#
# Why this exists: in a restricted/proxied GPU environment (Modal's gVisor+nvproxy
# sandbox is the case this was found against), NVIDIA RM's own device enumeration can
# report far more GPUs than are actually openable as device nodes in this specific
# container -- observed on Modal as a 16-entry RM report vs. exactly 1 real
# /dev/nvidiaN node. tinygrad's DEV=":<idx>+NV" selects by *position* in RM's raw
# enumeration, not by minor number directly, so the index has to be derived here.
#
# This relies on an assumption that is NOT a documented driver contract, only an
# empirically observed pattern (5/5 samples on Modal A100 containers): RM's list is
# gap-free and ascending, so a device node's minor number equals its position in that
# list. If that ever stops holding on some platform, this picks the wrong GPU instead
# of failing loudly -- if you need it to be robust to that instead, don't use this
# script; use a tinygrad build that reconciles by matching minor_number values
# directly (what this repo's NV backend did before this script replaced it).
#
# The minor number itself changes on every container run/redeploy (observed 3, 7, 1,
# 8, 6 across 5 separate Modal launches) -- it is never safe to hardcode, which is why
# this has to run fresh at job-launch time rather than being a fixed env var.
set -euo pipefail

minors=()
for dev in /dev/nvidia[0-9]*; do
  [ -e "$dev" ] || continue
  minors+=("${dev##*/dev/nvidia}")
done

if [ "${#minors[@]}" -eq 0 ]; then
  echo "modal_select_gpu.sh: no /dev/nvidiaN device nodes found in this container" >&2
  exit 1
fi

IFS=,
export DEV=":${minors[*]}+NV"
unset IFS

exec "$@"
