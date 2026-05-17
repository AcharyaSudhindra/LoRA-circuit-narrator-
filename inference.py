from pathlib import Path
from peft import PeftModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load base model and merge LoRA adapter from local results folder
base_model_name = "EleutherAI/gpt-neo-125M"
model_path = Path(__file__).resolve().parent / "results"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, str(model_path)).merge_and_unload()

# Create pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt
prompt = "Explain a flip-flop in a story"

# Generate output
output = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])
