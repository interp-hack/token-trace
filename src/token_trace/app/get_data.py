from hashlib import md5
from pathlib import Path

import pandas as pd
from token_trace.compute_node_attribution import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXT,
    compute_node_attribution,
)

DATA_DIR = Path("data")


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
    return df


def load_or_compute_data(text: str) -> pd.DataFrame:
    # Load or compute node attributions
    hash = md5(text.encode()).hexdigest()[:16]
    filepath = DATA_DIR / f"{hash}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0)
    else:
        # Compute node attributions
        df = compute_node_attribution(DEFAULT_MODEL_NAME, text)
        df.to_csv(filepath)

    return df


def get_data(text: str) -> pd.DataFrame:
    df = load_or_compute_data(text)
    return process_data(df)


if __name__ == "__main__":
    get_data(DEFAULT_TEXT)
