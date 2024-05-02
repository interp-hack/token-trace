import pathlib

from token_trace.app import run_app

data_dir = pathlib.Path(__file__).parent / "data"

if __name__ == "__main__":
    # For now, restrict to precomputed circuits only in deployment
    run_app(precomputed_only=True)
