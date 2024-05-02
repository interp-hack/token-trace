from collections.abc import Sequence
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import streamlit as st
from annotated_text import annotated_text
from pandera.typing import Series
from plotly.subplots import make_subplots

from token_trace.app.get_circuit import get_circuit, list_existing_circuits
from token_trace.app.process_data import process_node_data
from token_trace.constants import (
    # DEFAULT_ANSWER,
    DEFAULT_MODEL_NAME,
    # DEFAULT_PROMPT,
    DEFAULT_REPO_ID,
    DEFAULT_TEXT,
)
from token_trace.load_pretrained_model import load_model
from token_trace.print_prompt_info import print_prompt_info
from token_trace.utils import get_neuronpedia_url, get_token_strs

pd.options.mode.chained_assignment = None  # default='warn'


def get_token_annotations(tokens: list[str]) -> Sequence[str | tuple[str, str, str]]:
    """Helper to indicate which token is being considered."""

    # TODO: increase font size?
    second_last_token_annotation = (tokens[-2], "loss", "#ffa421")
    last_token_annotation = (tokens[-1], "label", "#0F52BA")
    return [token for token in tokens[:-2]] + [
        second_last_token_annotation,
        last_token_annotation,
    ]


def plot_indirect_effect_vs_activation(df: pd.DataFrame):
    df = df[df["act_type"] == "feature"]
    fig = px.scatter(
        df,
        x="value",
        y="ie",
        color="layer",
        title="Indirect effect vs activation",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def add_section_total_attribution(df: pd.DataFrame):
    st.header("Summary of Feature Attribution")

    total_ie_df = df[
        ["layer", "act_type", "total_abs_ie_by_layer_and_act_type"]
    ].drop_duplicates()
    total_ie_df["name"] = "const"
    total_ie_df = total_ie_df.sort_values(
        ["layer", "total_abs_ie_by_layer_and_act_type"], ascending=[True, False]
    )

    left, right = st.columns(2)
    with left:
        # Bar chart of total attribution by layer
        fig = px.bar(
            total_ie_df,
            x="layer",
            y="total_abs_ie_by_layer_and_act_type",
            color="act_type",
            title="Total node attributions by layer",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        # Pie chart of total attribution by node type
        df["total_abs_ie_by_act_type"] = df.groupby("act_type")["abs_ie"].transform(
            "sum"
        )
        pie_df = df[["act_type", "total_abs_ie_by_act_type"]].drop_duplicates()
        fig = px.pie(
            pie_df,
            values="total_abs_ie_by_act_type",
            names="act_type",
            title="Total attribution by node type",
            color="act_type",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )


def plot_bar_frac_total_abs_ie_by_layer_and_act_type(df: pd.DataFrame):
    # Filter by act_type = feature
    df = df[df.act_type == "feature"]
    df = df.sort_values(
        ["layer", "frac_total_abs_ie_by_layer_and_act_type"], ascending=[True, False]
    )

    fig = px.bar(
        df,
        x="frac_total_abs_ie_by_layer_and_act_type",
        y="layer",
        text="act_idx",
        title="Fraction of total attribution within layer",
        color="ie",
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
        label="Select the top K nodes to open in Neuronpedia",
        # Log scale slider
        options=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        value=100,
    )

    positive, negative = st.columns(2)
    assert isinstance(k_nodes, int)
    df = df.sort_values(
        ["total_abs_ie_across_token_position", "layer"], ascending=[False, True]
    )
    df = df.head(k_nodes)

    with positive:
        # NOTE: Seems like a pain to do on-click events.
        # Gonna settle for "open in neuronpedia" button.
        pos_df = df[df["ie"] > 0]
        for layer in pos_df["layer"].sort_values().unique():
            features = pos_df[pos_df["layer"] == layer]["act_idx"].values
            list_name = f"layer_{layer}_ie_positive_features"
            st.link_button(
                label=f"ie-positive features for layer {layer}",
                url=get_neuronpedia_url(layer, features, list_name),
            )

    with negative:
        neg_df = df[df["ie"] < 0]
        for layer in neg_df["layer"].sort_values().unique():
            features = neg_df[neg_df["layer"] == layer]["act_idx"].values
            list_name = f"layer_{layer}_ie_negative_features"
            st.link_button(
                label=f"ie-negative features for layer {layer}",
                url=get_neuronpedia_url(layer, features, list_name),
            )


def add_section_individual_feature_attribution(df: pd.DataFrame):
    st.header("Individual Feature Attributions")
    st.write("Here, we visualize the feature attributions for each node.")

    df = df[df["act_type"] == "feature"]

    left, right = st.columns(2)
    with left:
        st.header("Fraction of Total Attribution by Layer")
        st.write("Here, we visualize the fraction of total attribution by layer.")
        plot_bar_frac_total_abs_ie_by_layer_and_act_type(df)

    with right:
        st.header("Open in NeuronPedia")
        add_neuronpedia_buttons(df)


def plot_tokenwise_feature_attribution_for_layer(
    df: pd.DataFrame,
    layer: int,
    tokens: list[str],
    title: str = "Indirect effect by token position",
):
    def get_ie_df_for_layer_and_feature(df: pd.DataFrame, layer: int, feature: int):
        df = df[(df["layer"] == layer) & (df["act_idx"] == feature)]
        indirect_effects = df[["ie", "token_idx", "layer", "act_idx"]]
        # Create a combined "layer_and_act_idx" column
        indirect_effects["layer_and_act_idx"] = indirect_effects.apply(
            lambda row: f"({int(row['layer'])}, {int(row['act_idx'])})", axis=1
        )
        # Impute missing tokens
        missing_rows = []
        for token_idx, _ in enumerate(tokens):
            if token_idx not in indirect_effects["token_idx"].values:
                missing_rows.append(
                    {
                        "token_idx": token_idx,
                        "ie": 0,
                        "layer": layer,
                        "act_idx": feature,
                        "layer_and_act_idx": f"({layer}, {feature})",
                    }
                )
        missing_rows_df = pd.DataFrame(missing_rows)
        indirect_effects = pd.concat([indirect_effects, missing_rows_df])
        indirect_effects["token_str"] = indirect_effects["token_idx"].apply(
            lambda idx: tokens[idx]
        )
        return indirect_effects

    # Set up layers, features to visualize
    # DEFAULT: pick top 10 features from specified layer

    k_nodes = 10

    def get_top_k_features(df: pd.DataFrame, layer: int, k_nodes: int) -> Series[int]:
        df = df[
            ["layer", "act_idx", "act_type", "total_abs_ie_across_token_position"]
        ].drop_duplicates()
        df = df[df["act_type"] == "feature"]
        df = df[df["layer"] == layer]
        df = df.sort_values(
            ["total_abs_ie_across_token_position", "layer"], ascending=[False, True]
        )
        top_k = df.head(k_nodes)
        features = top_k[top_k["layer"] == layer]["act_idx"]

        return cast(Series[int], features)

    features = get_top_k_features(df, layer, k_nodes)

    layers_and_features = [(layer, feature) for feature in features]

    dfs = []
    for layer, feature_idx in layers_and_features:
        dfs.append(get_ie_df_for_layer_and_feature(df, int(layer), int(feature_idx)))

    if len(dfs) > 0:
        indirect_effects = pd.concat(dfs)

        # Bar plot of indirect effect by token position
        fig = px.bar(
            indirect_effects,
            y="token_idx",
            x="ie",
            color="layer_and_act_idx",
            title=title,
            orientation="h",
        )
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(tokens))),
                ticktext=tokens,
                autorange="reversed",
            )
        )
        return fig

    else:
        return None


def add_section_tokenwise_all_layers(tokens: list[str], df: pd.DataFrame):
    st.header("Tokenwise Feature Attributions for All Layers")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        for layer in [0, 4, 8]:
            subfig = plot_tokenwise_feature_attribution_for_layer(
                df,
                layer,
                tokens,
                title=f"Layer {layer}: Indirect effect by token position",
            )
            st.plotly_chart(subfig, use_container_width=True)
    with col2:
        for layer in [1, 5, 9]:
            subfig = plot_tokenwise_feature_attribution_for_layer(
                df,
                layer,
                tokens,
                title=f"Layer {layer}: Indirect effect by token position",
            )
            st.plotly_chart(subfig, use_container_width=True)
    with col3:
        for layer in [2, 6, 10]:
            subfig = plot_tokenwise_feature_attribution_for_layer(
                df,
                layer,
                tokens,
                title=f"Layer {layer}: Indirect effect by token position",
            )

            st.plotly_chart(subfig, use_container_width=True)
    with col4:
        for layer in [3, 7, 11]:
            subfig = plot_tokenwise_feature_attribution_for_layer(
                df,
                layer,
                tokens,
                title=f"Layer {layer}: Indirect effect by token position",
            )
            st.plotly_chart(subfig, use_container_width=True)


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


def run_app(precomputed_only: bool = True):
    st.set_page_config(layout="wide")
    # Display model name
    st.header("Metadata")
    st.write(f"Model: {DEFAULT_MODEL_NAME}")
    st.write(f"SAEs: {DEFAULT_REPO_ID}")

    # List existing circuits
    existing_texts = list_existing_circuits() + ["null"]
    st.header("View a pre-computed prompt")
    selected_text = st.selectbox("Select a prompt", existing_texts, index=0)
    text = selected_text
    assert isinstance(text, str)

    # Get text
    if not precomputed_only:
        st.header("Or write your own prompt")
        user_text = st.text_input("Text", DEFAULT_TEXT)
        text = selected_text if selected_text else user_text

    st.divider()
    prompt, response = text.rsplit(" ", 1)

    with st.expander("Prompt breakdown"):
        # Display tokenized text
        st.write("Tokenized text:")
        tokens = get_token_strs(DEFAULT_MODEL_NAME, text)
        annotated_tokens = get_token_annotations(tokens)
        annotated_text(*annotated_tokens)

        # Display prompt_info
        model = load_model(DEFAULT_MODEL_NAME)
        print_prompt_info(prompt, response, model, print_fn=st.write)

    # Load or compute node attributions
    circuit = get_circuit(text)
    df = process_node_data(circuit.node_ie_df)

    # TODO: Summarise SAE errors
    add_section_total_attribution(df.copy())
    plot_indirect_effect_vs_activation(df.copy())
    add_section_individual_feature_attribution(df.copy())
    add_section_tokenwise_all_layers(tokens, df.copy())
