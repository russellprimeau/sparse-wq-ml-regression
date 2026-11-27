"""Portable locations for the study workflow."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"config"
STUDY_YML=CONFIG/"study.yml"
RAW=ROOT/"data/raw"
DERIVED=ROOT/"data/derived"
RESULTS=ROOT/"results"
OUTPUTS=ROOT/"outputs"
