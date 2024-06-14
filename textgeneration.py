from transformers import pipeline, set_seed

# Set random seed for reproducibility (optional)
set_seed(42)

# Load the text generation pipeline
generator = pipeline("text-generation", model="gpt2")  

# Define the starting text (prompt)
prompt = "In a world where robots have become sentient,"

# Generation parameters
generation_params = {
    "max_length": 50,           # Maximum length of generated text
    "num_return_sequences": 3,  # Number of sequences to generate
}

# --- Greedy Search (Default) ---
print("\nGreedy Search:")
greedy_output = generator(prompt, **generation_params)
for i, output in enumerate(greedy_output):
    print(f"({i+1}) {output['generated_text']}")

# --- Beam Search ---
print("\nBeam Search:")
beam_output = generator(prompt, num_beams=5, early_stopping=True, **generation_params)
for i, output in enumerate(beam_output):
    print(f"({i+1}) {output['generated_text']}")

# --- Top-k Sampling ---
print("\nTop-k Sampling:")
topk_output = generator(prompt, do_sample=True, top_k=50, **generation_params)
for i, output in enumerate(topk_output):
    print(f"({i+1}) {output['generated_text']}")

# --- Top-p (Nucleus) Sampling ---
print("\nTop-p Sampling:")
topp_output = generator(prompt, do_sample=True, top_p=0.92, **generation_params)
for i, output in enumerate(topp_output):
    print(f"({i+1}) {output['generated_text']}")
