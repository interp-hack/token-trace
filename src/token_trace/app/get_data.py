import pathlib
from collections import deque
from hashlib import md5
from pathlib import Path
from threading import Lock

import pandas as pd
from token_trace.compute_node_attribution import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXT,
    compute_node_attribution,
)

DATA_DIR = Path("data")

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
        df.groupby(["layer", "node_type"])["abs_ie"]
        .sum()
        .rename("total_abs_ie_by_layer_and_node_type")
    )
    df = df.merge(total_abs_ie_by_layer_and_node_type, on=["layer", "node_type"])
    # Add fraction of total attribution within layer
    df["frac_total_abs_ie_by_layer_and_node_type"] = (
        df["abs_ie"] / df["total_abs_ie_by_layer_and_node_type"]
    )
    # Add layer as string
    df["layer_str"] = df["layer"].astype(str)
    # Add total absolute indirect effect across token position
    df["total_abs_ie_across_token_position"] = df.groupby(["layer", "feature"])[
        "abs_ie"
    ].transform("sum")

    return df


def load_or_compute_data(text: str, force_rerun: bool = False) -> pd.DataFrame:
    # Load or compute node attributions
    hash = md5(text.encode()).hexdigest()[:16]
    filepath = DATA_DIR / f"{hash}.csv"
    if filepath.exists() and not force_rerun:
        df = pd.read_csv(filepath, index_col=0)
    else:
        # Compute node attributions
        df = compute_node_attribution(DEFAULT_MODEL_NAME, text)
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
