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
grounded_prompt = (
    "You explain digital circuits in very simple terms. "
    "Stay factual. If unsure, say you are unsure.\n"
    f"Q: {prompt.strip()}\nA:"
)

# Generate output
output = generator(
    grounded_prompt,
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
print(output[0]["generated_text"].split("A:")[-1].strip())
