from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from typing import cast
from trl.trainer.sft_trainer import SFTTrainer
import torch

# Base model identifier from Hugging Face Hub
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Load the custom training dataset from a local JSONL file
dataset = load_dataset("json", data_files="./dataset.jsonl")

# Function to format data into the ChatML template expected by the Qwen model
def format_data(data):
    return f"""<|im_start|>system
Answer the question about PyPDFForm.<|im_end|>
<|im_start|>user
Instruction:
{data['instruction']}

Input:
{data['input']}<|im_end|>
<|im_start|>assistant
{data['output']}<|im_end|><|endoftext|>"""

# Load the tokenizer associated with the base model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Set the padding token to be the same as the end-of-sequence token for consistent batching
tokenizer.pad_token = tokenizer.eos_token

# Configuration for 4-bit quantization (QLoRA) to reduce memory usage during training
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Enable 4-bit loading
    bnb_4bit_compute_dtype=torch.bfloat16,   # Computation happens in bfloat16 for better performance
    bnb_4bit_quant_type="nf4",               # Use Normalized Float 4 (optimized for weights)
    bnb_4bit_use_double_quant=True,          # Quantize the quantization constants for further savings
)

# Load the base Causal Language Model with quantization and automatic device placement (e.g., GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare the quantized model for Parameter-Efficient Fine-Tuning (PEFT)
model = prepare_model_for_kbit_training(model)

# Configuration for LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8,                                     # Rank of the adaptation (lower rank = fewer parameters)
    lora_alpha=16,                           # Scaling factor for the LoRA weights
    # List of target layers to apply LoRA adapters to
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,                       # Dropout rate for LoRA layers to prevent overfitting
    task_type="CAUSAL_LM"                    # Task type is Causal Language Modeling
)

# Wrap the base model with LoRA adapters based on the configuration
model = get_peft_model(model, lora_config)

# Define hyperparameters and settings for the training loop
training_args = TrainingArguments(
    output_dir="./model",                    # Directory where model checkpoints will be saved
    per_device_train_batch_size=2,           # Number of samples per GPU for each training step
    gradient_accumulation_steps=4,           # Number of steps to accumulate gradients before updating
    num_train_epochs=3,                      # Total number of times to iterate over the dataset
    learning_rate=2e-4,                      # Initial learning rate for the optimizer
    logging_steps=10,                        # Frequency of logging training metrics
    save_strategy="epoch",                   # Save a checkpoint at the end of every epoch
    dataloader_pin_memory=False,             # Disable pinned memory if running into resource issues
    bf16=True,                               # Use Brain Floating Point 16 (highly recommended for modern GPUs)
)

# Initialize the Supervised Fine-Tuning (SFT) trainer
trainer = SFTTrainer(
    model=cast(PeftModel, model),            # The model with LoRA adapters
    train_dataset=dataset["train"],          # The training split of our dataset
    args=training_args,                      # Training hyperparameters
    formatting_func=format_data,             # Function to convert raw data into prompt strings
)

# Execute the fine-tuning process
trainer.train()
