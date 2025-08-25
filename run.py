import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from ant_colony.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
