import pathlib
from collections import deque
from hashlib import md5
from threading import Lock

import pandas as pd

from token_trace.circuit import (
    compute_node_attribution,
)
from token_trace.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXT,
)
from token_trace.load_pretrained_model import load_model, load_sae_dict
from token_trace.utils import last_token_prediction_loss

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "app" / "data"
# Maximum number of files allowed
MAX_FILES = 10_000
FILE_QUEUE: deque[pathlib.Path] = deque()


def add_file_and_delete_old(file_path: pathlib.Path):
    # Add new file path to the queue
    FILE_QUEUE.append(file_path)
    # Check if the number of files exceeded the limit
    if len(FILE_QUEUE) > MAX_FILES:
        # Remove the oldest file
        oldest_file = FILE_QUEUE.popleft()
        oldest_file.unlink()
        print(f"Deleted old file: {oldest_file}")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to add additional columns."""
    # Add absolute indirect effect
    df["abs_ie"] = df["indirect_effect"].abs()
    # Add total absolute indirect effect in layer
    total_abs_ie_by_layer_and_node_type = (
        df.groupby(["example_idx", "layer", "node_type"])["abs_ie"]
        .sum()
        .rename("total_abs_ie_by_layer_and_node_type")
    )
    df = df.merge(
        total_abs_ie_by_layer_and_node_type, on=["example_idx", "layer", "node_type"]
    )
    # Add fraction of total attribution within layer
    df["frac_total_abs_ie_by_layer_and_node_type"] = (
        df["abs_ie"] / df["total_abs_ie_by_layer_and_node_type"]
    )
    # Add layer as string
    df["layer_str"] = df["layer"].astype(str)
    # Add total absolute indirect effect across token position
    df["total_abs_ie_across_token_position"] = df.groupby(
        ["example_idx", "layer", "node_idx"]
    )["abs_ie"].transform("sum")

    return df


def load_or_compute_data(text: str, force_rerun: bool = False) -> pd.DataFrame:
    # Load or compute node attributions
    DATA_DIR.mkdir(exist_ok=True)
    hash = md5(text.encode()).hexdigest()[:16]
    filepath = DATA_DIR / f"{hash}.csv"
    if filepath.exists() and not force_rerun:
        df = pd.read_csv(filepath, index_col=0)
    else:
        # Compute node attributions
        model = load_model(DEFAULT_MODEL_NAME)
        sae_dict = load_sae_dict(DEFAULT_MODEL_NAME)
        metric_fn = last_token_prediction_loss
        df = compute_node_attribution(model, sae_dict, metric_fn, text)
        df.to_csv(filepath)
        add_file_and_delete_old(filepath)

    return df


def get_data(text: str, force_rerun: bool = False) -> pd.DataFrame:
    mutex = Lock()
    with mutex:
        df = load_or_compute_data(text, force_rerun)
    return process_data(df)


if __name__ == "__main__":
    get_data(DEFAULT_TEXT, force_rerun=True)
