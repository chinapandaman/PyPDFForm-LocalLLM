import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "microsoft/phi-2"
adapter_path = "./model/checkpoint-303"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto"
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()

def execute(p):
    prompt = f"""Answer the question about PyPDFForm.

Question:
{p}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.2,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Answer:")[-1].strip()


while True:
    text = input("\nEnter question: ")
    print("\n", execute(text))
