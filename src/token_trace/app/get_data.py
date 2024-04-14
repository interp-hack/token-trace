from hashlib import md5
from pathlib import Path

import pandas as pd
from token_trace.compute_node_attribution import (
    DEFAULT_MODEL_NAME,
    compute_node_attribution,
)

DATA_DIR = Path("data")


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to add additional columns."""
    # Add absolute indirect effect
    df["abs_ie"] = df["indirect_effect"].abs()
    total_abs_ie_in_layer = (
        df.groupby("layer")["abs_ie"].sum().rename("total_abs_ie_in_layer")
    )
    df = df.merge(total_abs_ie_in_layer, on="layer")
    # Add fraction of total attribution within layer
    df["frac_total_abs_ie_in_layer"] = df["abs_ie"] / df["total_abs_ie_in_layer"]
    # Add layer as string
    df["layer_str"] = df["layer"].astype(str)
    return df


def get_data(text: str) -> pd.DataFrame:
    # Load or compute node attributions
    hash = md5(text.encode()).hexdigest()[:16]
    filepath = DATA_DIR / f"{hash}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0)
    else:
        # Compute node attributions
        df = compute_node_attribution(DEFAULT_MODEL_NAME, text)
        df.to_csv(filepath)

    # Process the dataframe
    df = process_df(df)
    return df
