from collections.abc import Sequence
from hashlib import md5
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import streamlit as st
from annotated_text import annotated_text
from plotly.subplots import make_subplots
from token_trace.compute_node_attribution import (
    DEFAULT_MODEL_NAME,
    DEFAULT_REPO_ID,
    DEFAULT_TEXT,
    compute_node_attribution,
    get_token_strs,
)
from token_trace.utils import open_neuronpedia

DATA_DIR = Path("data")


def get_token_annotations(tokens: list[str]) -> Sequence[str | tuple[str, str, str]]:
    """Helper to indicate which token is being considered."""

    last_token_annotation = (tokens[-1], "loss", "#ffa421")
    return [token for token in tokens[:-1]] + [last_token_annotation]


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_ie"] = df["indirect_effect"].abs()
    total_abs_ie_in_layer = (
        df.groupby("layer")["abs_ie"].sum().rename("total_abs_ie_in_layer")
    )
    # Merge
    df = df.merge(total_abs_ie_in_layer, on="layer")
    df["frac_total_abs_ie_in_layer"] = df["abs_ie"] / df["total_abs_ie_in_layer"]
    df["layer_str"] = df["layer"].astype(str)
    return df


def plot_total_attribution(df: pd.DataFrame):
    total_ie_df = df[["layer", "layer_str", "total_abs_ie_in_layer"]].drop_duplicates()
    total_ie_df["name"] = "const"
    total_ie_df = total_ie_df.sort_values(
        ["layer", "total_abs_ie_in_layer"], ascending=[True, False]
    )
    fig = px.pie(
        total_ie_df,
        values="total_abs_ie_in_layer",
        names="layer_str",
        title="Total attribution by layer",
        color="layer_str",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_feature_attribution_bar(df: pd.DataFrame):
    left, right = st.columns(2)
    with left:
        df = df.sort_values(
            ["layer", "frac_total_abs_ie_in_layer"], ascending=[True, False]
        )

        fig = px.bar(
            df,
            x="frac_total_abs_ie_in_layer",
            y="layer",
            text="feature",
            title="Fraction of total attribution within layer",
            color="indirect_effect",
            color_continuous_scale=px.colors.diverging.Fall_r,
            color_continuous_midpoint=0,
            orientation="h",
        )
        fig.update_layout(xaxis_range=[0, 1])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(height=800)
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        positive, negative = st.columns(2)
        with positive:
            # NOTE: Seems like a pain to do on-click events.
            # Gonna settle for "open in neuronpedia" button.
            pos_df = df[df["indirect_effect"] > 0]
            for layer in pos_df["layer"].sort_values().unique():
                features = pos_df[pos_df["layer"] == layer]["feature"].values
                list_name = f"layer_{layer}_positive_features"
                st.button(
                    f"Positive features for layer {layer}",
                    on_click=open_neuronpedia,
                    args=(layer, features, list_name),
                )

        with negative:
            neg_df = df[df["indirect_effect"] < 0]
            for layer in neg_df["layer"].sort_values().unique():
                features = neg_df[neg_df["layer"] == layer]["feature"].values
                list_name = f"layer_{layer}_negative_features"
                st.button(
                    f"Negative features for layer {layer}",
                    on_click=open_neuronpedia,
                    args=(layer, features),
                    kwargs={"name": list_name},
                )


def visualize_dataframe(df: pd.DataFrame):
    # Layer-specific stuff
    layers = df["layer"].unique()
    layers.sort()

    # Make a grid of plots
    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=[f"Layer {layer}" for layer in layers],
    )

    for layer in layers:
        # Calculate index
        layer_idx = int(layer)
        row = (layer_idx // 4) + 1
        col = (layer_idx % 4) + 1
        layer_df = df[df["layer"] == layer]
        # TODO: what other plots can we make?

        subfig = go.Histogram(x=layer_df.indirect_effect, name=f"Layer {layer}")
        fig.add_trace(subfig, row=row, col=col)

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # Display model name
    st.header("Metadata")
    st.write(f"Model: {DEFAULT_MODEL_NAME}")
    st.write(f"SAEs: {DEFAULT_REPO_ID}")

    # Get text
    st.header("Input")
    text = st.text_input("Enter a prompt: ", DEFAULT_TEXT)
    st.divider()

    # Display tokenized text
    st.header("Tokenized Text")
    tokens = get_token_strs(DEFAULT_MODEL_NAME, text)
    annotated_tokens = get_token_annotations(tokens)
    annotated_text(*annotated_tokens)

    # Load or compute node attributions
    hash = md5(text.encode()).hexdigest()[:16]
    filepath = DATA_DIR / f"{hash}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0)
    else:
        # Compute node attributions
        df = compute_node_attribution(DEFAULT_MODEL_NAME, text)
        df.to_csv(filepath)
    df = process_df(df)

    # Visualize the total attribution by layer.
    plot_total_attribution(df)

    # Select the top K nodes
    st.header("Feature Attribution")
    k_nodes = st.select_slider(
        label="Number of nodes",
        # Log scale slider
        options=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        value=100,
    )
    assert isinstance(k_nodes, int)

    # Load the data
    df = df.head(k_nodes)

    plot_feature_attribution_bar(df)
    visualize_dataframe(df)
