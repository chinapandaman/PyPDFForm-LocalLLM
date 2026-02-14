from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from typing import cast
from trl.trainer.sft_trainer import SFTTrainer

model_name = "microsoft/phi-2"

dataset = load_dataset("json", data_files="./dataset.jsonl")
def format_data(data):
    return f"""Answer the question about PyPDFForm.

Instruction:
{data['instruction']}

Input:
{data['input']}

Output:
{data['output']}"""

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    dataloader_pin_memory=False
)

trainer = SFTTrainer(
    model=cast(PeftModel, model),
    train_dataset=dataset["train"],
    args=training_args,
    formatting_func=format_data,
)

trainer.train()
