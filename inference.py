from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model directly from local folder
model_path = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt
prompt = "Explain a flip-flop in a story"

# Generate output
output = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])