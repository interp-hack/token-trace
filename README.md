# Token-Trace

A tool and UI to construct prompt-centric views of SAE feature attributions.

Main functionality:
- We use this tool to identify which SAE features have the most 'attribution' towards decreasing the model loss. 
- In combination with Neuronpedia, we can identify what each SAE feature represents; this then gives us a rough idea of what computation the model is performing. 

This tool is a first step towards discovering information flow between the features / layers of a transformer


# Quickstart

## Installation

```
pip install token-trace
```
## Example Usage

```
from token_trace import compute_node_attribution

text = "When John and Mary went to the shops, John gave the bag to Mary"

df: pd.DataFrame = compute_node_attribution(
    model_name = "gpt2",
    text
)
```

Each row of `df` describes one node corresponding to an SAE feature or error term. 

## Visualizing SAE attribution statistics in frontend. 

We use [Streamlit](https://streamlit.io/) to create a UI. Start the app as follows:
```
streamlit run app/token_trace_app.py
```

## Methodology

Under the hood, we use attribution patching to compute indirect effect of the loss with respect to SAE features. The method is adapted heavily from [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647). 

## Development

We use [PDM](https://github.com/pdm-project/pdm) to manage dependencies. Set up a development environment as follows:
```
pdm install # creates a .venv
source .venv/bin/activate
```

Once in the virtual environment, make sure to also install the pre-commit hooks
```
pre-commit install
```