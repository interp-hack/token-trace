from token_trace.app import run_app

if __name__ == "__main__":
    # For now, restrict to precomputed circuits only in deployment
    run_app(precomputed_only=True)
