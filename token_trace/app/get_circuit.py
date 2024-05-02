import json
import pathlib
import shutil
from collections import deque
from hashlib import md5
from threading import Lock

from token_trace.circuit import SparseFeatureCircuit, SparseFeatureCircuitBuilder
from token_trace.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXT,
)

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "app" / "data"
# Maximum number of files allowed
MAX_FILES = 10_000
PATH_QUEUE: deque[pathlib.Path] = deque()


def add_path_and_delete_old(path: pathlib.Path):
    # Add new file path to the queue
    PATH_QUEUE.append(path)
    # Check if the number of files exceeded the limit

    if len(PATH_QUEUE) > MAX_FILES:
        # Remove the oldest file
        oldest_path = PATH_QUEUE.popleft()
        if oldest_path.exists():
            if oldest_path.is_dir():
                # Remove the directory and its contents
                shutil.rmtree(oldest_path)
            else:
                oldest_path.unlink()
        print(f"Deleted old path: {oldest_path}")


def list_existing_circuits() -> list[str]:
    existing_texts = []
    savedirs = [path for path in DATA_DIR.iterdir() if path.is_dir()]
    # get the text for each circuit
    for savedir in savedirs:
        try:
            with open(savedir / "args.json") as f:
                args = json.load(f)
                existing_texts.append(args["text"])
        except FileNotFoundError:
            continue
    return existing_texts


def load_or_compute_circuit(
    text: str, force_rerun: bool = False
) -> SparseFeatureCircuit:
    """Load or compute the circuit data."""
    prefix = md5(text.encode()).hexdigest()[:16]
    save_dir = DATA_DIR / prefix

    if save_dir.exists() and not force_rerun:
        circuit = SparseFeatureCircuit.load(DATA_DIR / prefix)
    else:
        save_dir.mkdir(exist_ok=True, parents=True)
        builder = SparseFeatureCircuitBuilder(model_name=DEFAULT_MODEL_NAME, text=text)
        # TODO: edge attributions are still too slow to compute this here
        builder.save_args(save_dir)
        builder.compute_sae_activation_cache().compute_node_attributions()
        circuit = builder.circuit
        circuit.save(save_dir)
        add_path_and_delete_old(save_dir)

    return circuit


def get_circuit(text: str, force_rerun: bool = False) -> SparseFeatureCircuit:
    mutex = Lock()
    with mutex:
        circuit = load_or_compute_circuit(text, force_rerun)
    return circuit


if __name__ == "__main__":
    get_circuit(DEFAULT_TEXT, force_rerun=True)
