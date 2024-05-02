from token_trace.app.get_circuit import load_or_compute_circuit

PROMPTS = [
    # IOI
    "When John and Mary went to the shops, John gave the bag to Mary",
    "When Tim and Jane went to the shops, Tim gave the bag to Jane",
    # Factual recall
    "Fact: Tokyo is a city in the country of Japan",
    "Fact: Delhi is a city in the country of India",
]

if __name__ == "__main__":
    for prompt in PROMPTS:
        circuit = load_or_compute_circuit(prompt)
        print(circuit)
