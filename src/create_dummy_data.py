import transformers
import pickle
import json
import torch
import random

from types import SimpleNamespace
from sae_lens.training.utils import BackwardsCompatibleUnpickler
from sae_lens import SparseAutoencoder
from huggingface_hub import hf_hub_download
from token_trace.utils import dump_jsonl, load_jsonl

# Hardcoded
model_name = "openai-community/gpt2"
prompt = "I am a human"
n_layers = 12
n_sae_features = 100 # 24576

def load_sae(layer: int) -> SparseAutoencoder:
    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    # Hacky way to get torch to unpickle an old version of SAELens model
    fake_pickle = SimpleNamespace()
    fake_pickle.Unpickler = BackwardsCompatibleUnpickler
    fake_pickle.__name__ = pickle.__name__
    data = torch.load(path, map_location=torch.device("cpu"), pickle_module=fake_pickle)
    sparse_autoencoder = SparseAutoencoder(cfg=data["cfg"])
    sparse_autoencoder.load_state_dict(data["state_dict"])
    return sparse_autoencoder

if __name__ == "__main__":
    # Load model and tokenizer 
    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"]
    token_strs = tokenizer.convert_ids_to_tokens(token_ids[0])
    
    # dummy_data: list of (token_str, layer, feature_index, feature_value)
    dummy_data = []
    for token_str in token_strs:
        for layer in range(n_layers):
            for feature_index in range(n_sae_features):
                rand_val = random.random()
                dummy_data.append((token_strs, layer, feature_index, rand_val))

    # Save dummy data
    dump_jsonl("dummy_data.jsonl", dummy_data)
