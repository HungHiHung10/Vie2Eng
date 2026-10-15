"""Run once online before starting Vie2Eng offline."""

from vie2eng.backends.ct2 import CT2Backend
from vie2eng.config import MODEL_SPECS


if __name__ == "__main__":
    backend = CT2Backend()
    for direction, spec in MODEL_SPECS.items():
        print(f"Preparing {spec['repo_id']}...", flush=True)
        backend.get_model(direction)
        print(f"Ready: {direction}", flush=True)
