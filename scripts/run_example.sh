
#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH="$(cd "$(dirname "$0")"/..; pwd)/src" python -m ant_colony.cli --random --n 8 --iters 125 --seed 123 --beta 3 --rho 0.45
