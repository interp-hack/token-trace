import pandas as pd

from token_trace.circuit.node_attribution import (
    compute_node_attribution,
    validate_node_attribution,
)
from token_trace.constants import DEFAULT_MODEL_NAME, DEFAULT_TEXT
from token_trace.load_pretrained_model import load_model, load_sae_dict
from token_trace.utils import last_token_prediction_loss

# schema for the node_df


class SparseFeatureCircuit:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        # TODO: support multiple text strings.
        text: str = DEFAULT_TEXT,
        min_node_ie: float = 0.02,
        min_edge_ie: float = 0.02,
    ):
        self.model_name = model_name
        self.text = text
        self.min_node_ie = min_node_ie
        self.min_edge_ie = min_edge_ie

        self.model = load_model(model_name)
        self.sae_dict = load_sae_dict(model_name)
        self.metric_fn = last_token_prediction_loss

    def num_nodes(self) -> int:
        return len(self.node_df)

    def compute_node_attributions(self) -> "SparseFeatureCircuit":
        # TODO: implement
        node_df = compute_node_attribution(
            model=self.model,
            sae_dict=self.sae_dict,
            metric_fn=self.metric_fn,
            text=self.text,
        )
        node_df = validate_node_attribution(node_df)
        self.node_df = node_df
        return self

    def filter_nodes(self) -> "SparseFeatureCircuit":
        # Filter nodes by total absolute indirect effect
        # (sum across token position, mean across examples)
        node_df = self.node_df
        node_df["absolute_indirect_effect"] = node_df["indirect_effect"].abs()
        node_df["total_absolute_indirect_effect_per_example"] = node_df.groupby(
            ["layer", "node_idx", "example_idx"]
        )["absolute_indirect_effect"].transform("sum")
        node_df["mean_absolute_indirect_effect"] = node_df.groupby(
            ["layer", "node_idx"]
        )["total_absolute_indirect_effect_per_example"].transform("mean")
        node_df = node_df[node_df["mean_absolute_indirect_effect"] > self.min_node_ie]
        node_df = validate_node_attribution(node_df)
        self.node_df = node_df
        return self

    def compute_edge_attributions(self) -> "SparseFeatureCircuit":
        return self

    def filter_edges(self) -> "SparseFeatureCircuit":
        return self

    def compute_circuit(self) -> "SparseFeatureCircuit":
        return (
            self.compute_node_attributions()
            .filter_nodes()
            .compute_edge_attributions()
            .filter_edges()
        )
