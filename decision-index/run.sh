#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$bundle_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT=https://huggingface.co
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$bundle_dir/.cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$bundle_dir/.cache/triton}"
python_bin="$bundle_dir/.venv/bin/python"
case "${1:-}" in
  smoke) exec "$python_bin" "$bundle_dir/smoke.py" "${@:2}" ;;
  text)
    # Run from the working directory containing the verified suite-0.3.
    exec "$python_bin" -m decision_index pipeline --engine spice_di:SpiceEngine \
      --option runtime=turbo --option model=yunqu/mentat-Spice-4B \
      --option revision=90d4f76971ebd07c11054ada274767458b615ad5 \
      --option device=cuda:0 "${@:2}" ;;
  vision)
    exec "$python_bin" -m decision_index vision pipeline --engine spice_di:SpiceEngine \
      --option runtime=turbo --option model=yunqu/mentat-Spice-4B \
      --option revision=90d4f76971ebd07c11054ada274767458b615ad5 \
      --option device=cuda:0 --workers 1 "${@:2}" ;;
  *) echo "Usage: bash run.sh smoke | text --out RUN | vision --dir SUITE --out RUN" >&2; exit 2 ;;
esac
