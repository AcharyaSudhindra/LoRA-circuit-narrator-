import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ── Check GPU ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    print(f" GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("  No GPU found — using CPU")

# ── 1. Load dataset ───────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files="dataset/circuit_story.jsonl", split="train")
print(f" Dataset loaded: {len(dataset)} examples")

# ── 2. Load base model + tokenizer ───────────────────────────────────────────
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.config.pad_token_id = tokenizer.eos_token_id

# ── 3. Preprocess dataset ─────────────────────────────────────────────────────
def preprocess_function(examples):
    inputs = [
        f"Q: {p}\nA: {r}"
        for p, r in zip(examples["prompt"], examples["response"])
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ── 4. Apply LoRA ─────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 5. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=50,
    save_strategy="epoch",
    learning_rate=5e-4,
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none",
    fp16=True if device == "cuda" else False,
    dataloader_pin_memory=True if device == "cuda" else False,
)

# ── 6. Trainer ────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print(f"\n Starting training on {device.upper()}...")
trainer.train()

# ── 8. Save ───────────────────────────────────────────────────────────────────
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
print(" Training complete. Model saved to ./results")