import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Check GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("No GPU found, using CPU")

# 1. Load dataset
dataset_path = Path(__file__).resolve().parent / "circuit_story.jsonl"
dataset = load_dataset("json", data_files=str(dataset_path), split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# 2. Load base model + tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.config.pad_token_id = tokenizer.eos_token_id

# 3. Preprocess dataset
# Train only on the answer tokens; mask prompt tokens with -100.
def preprocess_function(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for prompt, response in zip(examples["prompt"], examples["response"]):
        instruction = (
            "You explain digital circuits in very simple terms. "
            "Stay factual. If unsure, say you are unsure.\n"
            f"Q: {prompt.strip()}\nA:"
        )
        answer = f" {response.strip()}"

        prompt_tokens = tokenizer(instruction, add_special_tokens=False)
        answer_tokens = tokenizer(answer, add_special_tokens=False)

        input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_tokens["input_ids"])) + answer_tokens["input_ids"]

        max_len = 256
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + ([tokenizer.pad_token_id] * pad_len)
            attention_mask = attention_mask + ([0] * pad_len)
            labels = labels + ([-100] * pad_len)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4. Apply LoRA
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

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=8,
    save_strategy="epoch",
    learning_rate=2e-4,
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none",
    fp16=True if device == "cuda" else False,
    dataloader_pin_memory=True if device == "cuda" else False,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 7. Train
print(f"\nStarting training on {device.upper()}...")
trainer.train()

# 8. Save
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
print("Training complete. Model saved to ./results")
