import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen3-4B-Instruct-2507"
adapter_path = "./model/checkpoint-303" # Update this to your latest checkpoint, e.g., "./model/checkpoint-123"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)

print("Loading fine-tuned adapter...")
# Note: Ensure you have trained the model first and have a valid adapter in adapter_path
try:
    model = PeftModel.from_pretrained(model, adapter_path)
except Exception as e:
    print(f"Could not load adapter from {adapter_path}: {e}")
    print("Running with base model only.")

model.eval()

def execute(p):
    prompt = f"""<|im_start|>system
Answer the question about PyPDFForm.<|im_end|>
<|im_start|>user
{p}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return result.strip()


while True:
    text = input("\nEnter question: ")
    if text.lower() in ['exit', 'quit']:
        break
    print("\n", execute(text))
