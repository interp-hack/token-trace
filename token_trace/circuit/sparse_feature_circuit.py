import pandas as pd
import pandera as pa

from token_trace.circuit.node_attribution import compute_node_attribution

# schema for the node_df
node_df_schema = pa.DataFrameSchema(
    {
        "layer": pa.Column(int),
        "example_idx": pa.Column(int),
        "node_idx": pa.Column(int),
        "token_idx": pa.Column(int),
        "value": pa.Column(float),
        "grad": pa.Column(float),
        "indirect_effect": pa.Column(float),
        "node_type": pa.Column(str),
        "module_name": pa.Column(str),
        "token_str": pa.Column(str),
        "example_str": pa.Column(str),
    }
)


class SparseFeatureCircuit:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame

    def __init__(
        self,
        model_name: str = "gpt2",
        # TODO: support multiple text strings.
        text: str = "Hello, my name is",
        metric: str = "loss",
        min_node_ie: float = 0.02,
        min_edge_ie: float = 0.02,
    ):
        self.model_name = model_name
        self.text = text
        self.metric = metric
        self.min_node_ie = min_node_ie
        self.min_edge_ie = min_edge_ie

    def compute_node_attributions(self):
        node_df = compute_node_attribution(
            self.model_name,
            self.text,
            metric=self.metric,
        )
        node_df_schema.validate(node_df)
        self.node_df = node_df

    def filter_nodes(self):
        # Filter nodes by total absolute indirect effect
        # (sum across token position, mean across examples)
        node_df = self.node_df
        node_df["absolute_indirect_effect"] = node_df["indirect_effect"].abs()
        node_df["total_absolute_indirect_effect_per_example"] = node_df.groupby(
            ["layer", "node_idx", "example_idx"]
        )["absolute_indirect_effect"].transform("sum")
        node_df["mean_absolute_indirect_effect"] = node_df.groupby(
            ["layer", "node_idx"]
        )["absolute_indirect_effect_tokenwise_sum"].transform("mean")
        node_df = node_df[node_df["mean_absolute_indirect_effect"] > self.min_node_ie]
        node_df_schema.validate(node_df)
        self.node_df = node_df
        return self

    def compute_edge_attributions(self):
        pass

    def filter_edges(self):
        pass

    def compute_circuit(self):
        pass
