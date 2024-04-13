

```
from token_trace.utils import load_jsonl
data = load_jsonl("assets/dummy_data.jsonl")
# data is a list of (token_str, layer, feature_idx, feature_value) for all tokens, layers, feature indices
```