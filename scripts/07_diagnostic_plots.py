"""Article analysis: diagnostics"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_wq.workflow import stage_main
if __name__ == "__main__":
    stage_main('diagnostics')
