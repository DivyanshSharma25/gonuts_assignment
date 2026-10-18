import ollama

# Generate an embedding for a single text string
response = ollama.embed(
    model='nomic-embed-text',
    input='The sky is blue because of Rayleigh scattering',
)

# Access the generated list of float vectors
print(response.embeddings)
