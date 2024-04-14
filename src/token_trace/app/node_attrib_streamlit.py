from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import streamlit as st
from annotated_text import annotated_text
from plotly.subplots import make_subplots
from token_trace.app.get_data import get_data
from token_trace.compute_node_attribution import (
    DEFAULT_ANSWER,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_REPO_ID,
    get_token_strs,
)
from token_trace.utils import open_neuronpedia

DATA_DIR = Path("data")


def get_token_annotations(tokens: list[str]) -> Sequence[str | tuple[str, str, str]]:
    """Helper to indicate which token is being considered."""

    # TODO: increase font size?
    second_last_token_annotation = (tokens[-2], "loss", "#ffa421")
    last_token_annotation = (tokens[-1], "label", "#0F52BA")
    return [token for token in tokens[:-2]] + [
        second_last_token_annotation,
        last_token_annotation,
    ]


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


def add_section_total_attribution(df: pd.DataFrame):
    st.header("Summary of Feature Attribution")
    st.write("Here, we visualize the total feature attribution by layer.")

    total_ie_df = df[
        ["layer", "node_type", "layer_str", "total_abs_ie_by_layer_and_node_type"]
    ].drop_duplicates()
    total_ie_df["name"] = "const"
    total_ie_df = total_ie_df.sort_values(
        ["layer", "total_abs_ie_by_layer_and_node_type"], ascending=[True, False]
    )

    left, right = st.columns(2)
    with left:
        # Bar chart of total attribution by layer
        fig = px.bar(
            total_ie_df,
            x="layer_str",
            y="total_abs_ie_by_layer_and_node_type",
            color="node_type",
            title="Total node attributions by layer",
            # color="layer_str",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        # Pie chart of total attribution by node type
        df["total_abs_ie_by_node_type"] = df.groupby("node_type")["abs_ie"].transform(
            "sum"
        )
        pie_df = df[["node_type", "total_abs_ie_by_node_type"]].drop_duplicates()
        fig = px.pie(
            pie_df,
            values="total_abs_ie_by_node_type",
            names="node_type",
            title="Total attribution by node type",
            color="node_type",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # fig = px.pie(
    #     total_ie_df,
    #     values="total_abs_ie_in_layer",
    #     names="layer_str",
    #     title="Total attribution by layer",
    #     color="layer_str",
    # )
    # st.plotly_chart(
    #     fig,
    #     use_container_width=True,
    # )


def plot_bar_frac_total_abs_ie_by_layer_and_node_type(df: pd.DataFrame):
    # Filter by node_type = feature
    df = df[df["node_type"] == "feature"]
    df = df.sort_values(
        ["layer", "frac_total_abs_ie_by_layer_and_node_type"], ascending=[True, False]
    )

    fig = px.bar(
        df,
        x="frac_total_abs_ie_by_layer_and_node_type",
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


def add_neuronpedia_buttons(df: pd.DataFrame):
    # Select the top K nodes by total_abs_ie_across_token_position
    k_nodes = st.select_slider(
        label="Number of nodes",
        # Log scale slider
        options=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        value=100,
    )

    positive, negative = st.columns(2)

    # Compute sum of abs_ie across token position
    df["total_abs_ie_across_token_position"] = df.groupby(["layer", "feature"])[
        "abs_ie"
    ].transform("sum")

    assert isinstance(k_nodes, int)
    df = df.sort_values(
        ["total_abs_ie_across_token_position", "layer"], ascending=[False, True]
    )
    df = df.head(k_nodes)

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


def add_section_individual_feature_attribution(df: pd.DataFrame):
    st.header("Individual Feature Attributions")
    st.write("Here, we visualize the feature attributions for each node.")

    # Filter by node_type = feature
    df = df[df["node_type"] == "feature"]

    left, right = st.columns(2)
    with left:
        st.header("Fraction of Total Attribution by Layer")
        st.write("Here, we visualize the fraction of total attribution by layer.")
        plot_bar_frac_total_abs_ie_by_layer_and_node_type(df)

    with right:
        st.header("Open in NeuronPedia")
        add_neuronpedia_buttons(df)


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
    prompt = st.text_input("Enter a prompt: ", DEFAULT_PROMPT)
    response = st.text_input("Enter a response: ", DEFAULT_ANSWER)
    # pre-pend space to response if it doesn't already start with one
    if response and not response.startswith(" "):
        response = " " + response
    text = prompt + response
    st.divider()

    # Display tokenized text
    st.write("Tokenized text:")
    tokens = get_token_strs(DEFAULT_MODEL_NAME, text)
    annotated_tokens = get_token_annotations(tokens)
    annotated_text(*annotated_tokens)

    # Display test_prompt
    # model = load_model(DEFAULT_MODEL_NAME)
    # console = Console(record=True)
    # with console.capture() as capture:
    #     test_prompt(prompt, response, model, console = console)
    # st.markdown(console.export_html(), unsafe_allow_html=True)
    # print(console.export_html())
    # console.clear()

    # Load or compute node attributions
    df = get_data(text)

    add_section_total_attribution(df)
    add_section_individual_feature_attribution(df)
    # visualize_dataframe(df)
