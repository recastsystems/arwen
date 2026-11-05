"""Version-matched startup compiler images for gpuwm."""
from pathlib import Path

def data_path():
    return Path(__file__).resolve().parent / "data"
