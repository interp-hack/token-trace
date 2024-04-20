from token_trace.circuit.node_attribution import (
    NodeAttributionDataFrame,
    validate_node_attribution,
)


def process_node_data(node_df: NodeAttributionDataFrame) -> NodeAttributionDataFrame:
    """Process the node dataframe to add additional columns."""
    # Add total absolute indirect effect in layer
    total_abs_ie_by_layer_and_act_type = (
        node_df.groupby(["example_idx", "module_name", "act_type"])["abs_ie"]
        .sum()
        .rename("total_abs_ie_by_layer_and_act_type")
    )
    df = node_df.merge(
        total_abs_ie_by_layer_and_act_type,
        on=["example_idx", "module_name", "act_type"],
    )
    # Add fraction of total attribution within layer
    df["frac_total_abs_ie_by_layer_and_act_type"] = (
        df["abs_ie"] / df["total_abs_ie_by_layer_and_act_type"]
    )

    # Add total absolute indirect effect across token position
    df["total_abs_ie_across_token_position"] = df.groupby(
        ["example_idx", "module_name", "act_idx"]
    )["abs_ie"].transform("sum")

    return validate_node_attribution(df)
