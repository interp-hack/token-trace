import json
import pathlib
from dataclasses import dataclass

import pandas as pd

from token_trace.circuit.edge_attribution import (
    EdgeAttributionDataFrame,
    EdgeAttributionSchema,
    compute_edge_attribution,
    filter_edges,
    validate_edge_attribution,
)
from token_trace.circuit.node_attribution import (
    NodeAttributionDataFrame,
    NodeAttributionSchema,
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
from token_trace.utils import (
    get_empty_dataframe_from_pa_model,
    last_token_prediction_loss,
)

# schema for the node_df


class SparseFeatureCircuitBuilder:
    model: HookedTransformer
    sae_dict: SAEDict
    metric_fn: MetricFunction
    # TODO: support multiple text strings.
    text: str
    sae_activation_cache: SAEActivationCache

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        text: str = DEFAULT_TEXT,
        min_node_abs_ie: float = 0.0,
        max_n_nodes: int = -1,
        min_edge_abs_ie: float = 0.0,
        max_n_edges: int = -1,
    ):
        self.model_name = model_name
        self.text = text
        self.min_node_abs_ie = min_node_abs_ie
        self.min_edge_abs_ie = min_edge_abs_ie
        self.max_n_nodes = max_n_nodes
        self.max_n_edges = max_n_edges
        self.node_ie_df = None
        self.edge_ie_df = None

    @property
    def circuit(self):
        """Get the circuit at the current stage"""
        return SparseFeatureCircuit(
            node_ie_df=self.node_ie_df,
            edge_ie_df=self.edge_ie_df,
        )

    def compute_sae_activation_cache(self) -> "SparseFeatureCircuitBuilder":
        self.model = load_model(self.model_name)
        self.sae_dict = load_sae_dict(self.model_name)
        self.metric_fn = last_token_prediction_loss

        self.sae_activation_cache = get_sae_activation_cache(
            self.model, self.sae_dict, self.metric_fn, self.text
        )
        return self

    def compute_node_attributions(self) -> "SparseFeatureCircuitBuilder":
        self.node_ie_df = compute_node_attribution(
            model=self.model,
            sae_activation_cache=self.sae_activation_cache,
            text=self.text,
        )
        return self

    def get_filtered_nodes(
        self,
        min_node_abs_ie: float | None = None,
        max_n_nodes: int | None = None,
    ) -> NodeAttributionDataFrame:
        """Get filtered nodes by total absolute indirect effect"""
        if min_node_abs_ie is not None:
            self.min_node_abs_ie = min_node_abs_ie
        if max_n_nodes is not None:
            self.max_n_nodes = max_n_nodes
        assert self.node_ie_df is not None  # keep pyright happy
        return filter_nodes(
            self.node_ie_df,
            min_node_abs_ie=self.min_node_abs_ie,
            max_n_nodes=self.max_n_nodes,
        )

    def filter_nodes(
        self,
        min_node_abs_ie: float | None = None,
        max_n_nodes: int | None = None,
    ) -> "SparseFeatureCircuitBuilder":
        """Filter nodes by total absolute indirect effect"""
        self.node_ie_df = self.get_filtered_nodes(
            min_node_abs_ie=min_node_abs_ie,
            max_n_nodes=max_n_nodes,
        )
        return self

    def compute_edge_attributions(self) -> "SparseFeatureCircuitBuilder":
        assert self.node_ie_df is not None  # keep pyright happy
        self.edge_ie_df = compute_edge_attribution(
            self.node_ie_df,
            sae_acts_clean=self.sae_activation_cache,
        )
        return self

    def filter_edges(self) -> "SparseFeatureCircuitBuilder":
        assert self.edge_ie_df is not None  # keep pyright happy
        self.edge_ie_df = filter_edges(
            self.edge_ie_df,
            min_edge_ie=self.min_edge_abs_ie,
            max_n_edges=self.max_n_edges,
        )
        return self

    def compute_circuit(self) -> "SparseFeatureCircuitBuilder":
        return (
            self.compute_sae_activation_cache()
            .compute_node_attributions()
            .filter_nodes()
            .compute_edge_attributions()
            .filter_edges()
        )

    def save_args(self, save_dir: pathlib.Path):
        args = {
            "model_name": self.model_name,
            "text": self.text,
            "min_node_abs_ie": self.min_node_abs_ie,
            "max_n_nodes": self.max_n_nodes,
            "min_edge_abs_ie": self.min_edge_abs_ie,
            "max_n_edges": self.max_n_edges,
        }
        with open(save_dir / "args.json", "w") as f:
            json.dump(args, f)


@dataclass
class SparseFeatureCircuit:
    """Compute a circuit consisting of SAE features"""

    # Represent the sub-graph
    node_ie_df: NodeAttributionDataFrame
    edge_ie_df: EdgeAttributionDataFrame

    def __init__(
        self,
        node_ie_df: NodeAttributionDataFrame | None = None,
        edge_ie_df: EdgeAttributionDataFrame | None = None,
    ):
        if node_ie_df is None:
            node_ie_df = get_empty_dataframe_from_pa_model(NodeAttributionSchema)  # type: ignore
        if edge_ie_df is None:
            edge_ie_df = get_empty_dataframe_from_pa_model(EdgeAttributionSchema)  # type: ignore
        self.node_ie_df = validate_node_attribution(node_ie_df)  # type: ignore
        self.edge_ie_df = validate_edge_attribution(edge_ie_df)  # type: ignore

    """ Utility functions """

    def copy(self):
        return SparseFeatureCircuit(
            node_ie_df=self.node_ie_df.copy(),  # type: ignore
            edge_ie_df=self.edge_ie_df.copy(),  # type: ignore
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_ie_df)

    @property
    def num_edges(self) -> int:
        return len(self.edge_ie_df)

    def get_nodes_in_module(self, module_name: str) -> NodeAttributionDataFrame:
        return get_nodes_in_module(self.node_ie_df, module_name=module_name)

    """ Save and load """

    def save(self, save_dir: pathlib.Path):
        # Save results
        if hasattr(self, "node_ie_df"):
            self.node_ie_df.to_csv(save_dir / "node.csv")
        if hasattr(self, "edge_ie_df"):
            self.edge_ie_df.to_csv(save_dir / "edge.csv")

    @staticmethod
    def load(save_dir: pathlib.Path) -> "SparseFeatureCircuit":
        # Load results
        if (save_dir / "node.csv").exists():
            node_ie_df = pd.read_csv(save_dir / "node.csv", index_col=0)
        else:
            node_ie_df = None
        if (save_dir / "edge.csv").exists():
            edge_ie_df = pd.read_csv(save_dir / "edge.csv", index_col=0)
            if len(edge_ie_df) == 0:
                edge_ie_df = None
        else:
            edge_ie_df = None

        return SparseFeatureCircuit(
            node_ie_df=node_ie_df,  # type: ignore
            edge_ie_df=edge_ie_df,  # type: ignore
        )
