import json
import pathlib

import pandas as pd

from token_trace.circuit.edge_attribution import (
    EdgeAttributionDataFrame,
    compute_edge_attribution,
    filter_edges,
    validate_edge_attribution,
)
from token_trace.circuit.node_attribution import (
    NodeAttributionDataFrame,
    compute_node_attribution,
    filter_nodes,
    get_nodes_in_module,
    validate_node_attribution,
)
from token_trace.constants import DEFAULT_MODEL_NAME, DEFAULT_TEXT
from token_trace.load_pretrained_model import load_model, load_sae_dict
from token_trace.sae_activation_cache import (
    SAEActivationCache,
    get_sae_activation_cache,
)
from token_trace.types import (
    HookedTransformer,
    MetricFunction,
    SAEDict,
)
from token_trace.utils import last_token_prediction_loss

# schema for the node_df


class SparseFeatureCircuit:
    """Compute a circuit consisting of SAE features"""

    model: HookedTransformer
    sae_dict: SAEDict
    metric_fn: MetricFunction
    # TODO: support multiple text strings.
    text: str
    sae_activation_cache: SAEActivationCache

    # Represent the sub-graph
    node_ie_df: NodeAttributionDataFrame
    edge_ie_df: EdgeAttributionDataFrame

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        text: str = DEFAULT_TEXT,
        min_node_abs_ie: float = 0.0,
        max_n_nodes: int = 200,
        min_edge_abs_ie: float = 0.0,
        max_n_edges: int = -1,
    ):
        self.model_name = model_name
        self.text = text
        self.min_node_abs_ie = min_node_abs_ie
        self.min_edge_abs_ie = min_edge_abs_ie
        self.max_n_nodes = max_n_nodes
        self.max_n_edges = max_n_edges

    def compute_sae_activation_cache(self) -> "SparseFeatureCircuit":
        self.model = load_model(self.model_name)
        self.sae_dict = load_sae_dict(self.model_name)
        self.metric_fn = last_token_prediction_loss

        self.sae_activation_cache = get_sae_activation_cache(
            self.model, self.sae_dict, self.metric_fn, self.text
        )
        return self

    def compute_node_attributions(self) -> "SparseFeatureCircuit":
        # TODO: implement
        node_df = compute_node_attribution(
            model=self.model,
            sae_activation_cache=self.sae_activation_cache,
            text=self.text,
        )
        self.node_ie_df = node_df
        return self

    def filter_nodes(self) -> "SparseFeatureCircuit":
        """Filter nodes by total absolute indirect effect"""
        self.node_ie_df = filter_nodes(
            self.node_ie_df,
            min_node_abs_ie=self.min_node_abs_ie,
            max_n_nodes=self.max_n_nodes,
        )
        return self

    def compute_edge_attributions(self) -> "SparseFeatureCircuit":
        self.edge_ie_df = compute_edge_attribution(
            self.node_ie_df,
            sae_acts_clean=self.sae_activation_cache,
        )
        return self

    def filter_edges(self) -> "SparseFeatureCircuit":
        # TODO: implement
        self.edge_ie_df = filter_edges(
            self.edge_ie_df,
            min_edge_ie=self.min_edge_abs_ie,
            max_n_edges=self.max_n_edges,
        )
        return self

    def compute_circuit(self) -> "SparseFeatureCircuit":
        return (
            self.compute_sae_activation_cache()
            .compute_node_attributions()
            .filter_nodes()
            .compute_edge_attributions()
            .filter_edges()
        )

    """ Utility functions """

    @property
    def num_nodes(self) -> int:
        return len(self.node_ie_df)

    @property
    def num_edges(self) -> int:
        return len(self.edge_ie_df)

    def get_nodes_in_module(self, module_name: str) -> NodeAttributionDataFrame:
        return get_nodes_in_module(self.node_ie_df, module_name=module_name)

    """ Save and load """

    def save(self, save_dir: pathlib.Path, prefix: str = "circuit"):
        # Save constructor args
        with open(save_dir / (prefix + "_args.json"), "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "text": self.text,
                    "min_node_abs_ie": self.min_node_abs_ie,
                    "max_n_nodes": self.max_n_nodes,
                    "min_edge_abs_ie": self.min_edge_abs_ie,
                    "max_n_edges": self.max_n_edges,
                },
                f,
                indent=4,
            )

        # Save results
        self.node_ie_df.to_csv(save_dir / (prefix + "_node.csv"))
        self.edge_ie_df.to_csv(save_dir / (prefix + "_edge.csv"))

    @staticmethod
    def load(save_dir: pathlib.Path, prefix: str = "circuit") -> "SparseFeatureCircuit":
        with open(save_dir / (prefix + "_args.json")) as f:
            args = json.load(f)

        # Load results
        node_ie_df = pd.read_csv(save_dir / (prefix + "_node.csv"), index_col=0)
        edge_ie_df = pd.read_csv(save_dir / (prefix + "_edge.csv"), index_col=0)

        circuit = SparseFeatureCircuit(**args)
        circuit.node_ie_df = validate_node_attribution(node_ie_df)
        circuit.edge_ie_df = validate_edge_attribution(edge_ie_df)
        return circuit
