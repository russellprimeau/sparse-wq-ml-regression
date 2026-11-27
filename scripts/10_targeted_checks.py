"""Bounded sensitivity and calibration checks of fixed reported configurations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_wq.evaluate.targeted_checks import main
if __name__ == "__main__":
    main()
