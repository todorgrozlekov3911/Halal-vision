#!/usr/bin/env bash
# Halal-Vision environment setup (T01).
#
# Usage:
#   ./setup.sh            auto-detect machine (nvidia if nvidia-smi works, else intel)
#   ./setup.sh nvidia     force the NVIDIA stack
#   ./setup.sh intel      force the Intel (OpenVINO) stack
#
# Creates ./.venv, installs requirements/base.txt + the machine file, then
# verifies T01's "done when": every library imports and the accelerator is visible.

set -euo pipefail
cd "$(dirname "$0")"

CPU_TORCH_INDEX="https://download.pytorch.org/whl/cpu"

# --- pick machine profile -----------------------------------------------------
MACHINE="${1:-}"
if [[ -z "$MACHINE" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        MACHINE="nvidia"
    else
        MACHINE="intel"
    fi
    echo "Auto-detected machine profile: $MACHINE"
fi

REQ_FILE="requirements/${MACHINE}.txt"
if [[ ! -f "$REQ_FILE" ]]; then
    echo "Unknown machine '$MACHINE' (no $REQ_FILE). Use: nvidia | intel" >&2
    exit 1
fi

# --- venv ---------------------------------------------------------------------
if [[ ! -d .venv ]]; then
    echo "Creating virtualenv at .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# --- install ------------------------------------------------------------------
# One pip call so torch is resolved once: installing base.txt alone lets
# ultralytics pull the default CUDA torch (~4 GB of nvidia-* wheels) on Intel.
echo "Installing requirements/base.txt + $REQ_FILE"
if [[ "$MACHINE" == "intel" ]]; then
    pip install -r requirements/base.txt -r "$REQ_FILE" --extra-index-url "$CPU_TORCH_INDEX"
else
    pip install -r requirements/base.txt -r "$REQ_FILE"
fi

# --- verify (T01 "done when") ---------------------------------------------------
echo
echo "Verifying installation..."
python - "$MACHINE" <<'PY'
import sys

machine = sys.argv[1]
failed = False

for mod in ["numpy", "cv2", "ultralytics", "insightface", "yaml", "ffmpeg", "torch"]:
    try:
        __import__(mod)
        print(f"  import {mod:<12} OK")
    except Exception as e:
        print(f"  import {mod:<12} FAILED: {e}")
        failed = True

if machine == "nvidia":
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA GPU visible: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA GPU NOT visible (torch.cuda.is_available() is False)")
        failed = True
else:
    try:
        import openvino as ov
        devices = ov.Core().available_devices
        print(f"  OpenVINO devices: {devices}")
        if not any(d.startswith(("GPU", "NPU")) for d in devices):
            print("  WARNING: no iGPU/NPU visible to OpenVINO (CPU only).")
            print("  Check that intel-level-zero-gpu / NPU drivers are installed.")
            failed = True
    except Exception as e:
        print(f"  OpenVINO check FAILED: {e}")
        failed = True

sys.exit(1 if failed else 0)
PY

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"