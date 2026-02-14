import os
import re
import json
import random

def extract_sections(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by headers
    lines = content.split('\n')
    sections = []
    current_section = {"title": "", "text": "", "code_blocks": []}
    
    in_code_block = False
    current_code = []
    
    for line in lines:
        if line.startswith('```python'):
            in_code_block = True
            current_code = []
            continue
        elif line.startswith('```') and in_code_block:
            in_code_block = False
            current_section["code_blocks"].append("\n".join(current_code))
            continue
        
        if in_code_block:
            current_code.append(line)
            continue
            
        if line.startswith('#'):
            if current_section["title"] or current_section["text"] or current_section["code_blocks"]:
                sections.append(current_section)
            
            title = line.lstrip('#').strip()
            current_section = {"title": title, "text": "", "code_blocks": []}
        else:
            # Clean up mkdocs special syntax
            cleaned_line = re.sub(r'^=== ".*"$', '', line)
            cleaned_line = re.sub(r'^\s*\?\?\?\+.*$', '', cleaned_line)
            cleaned_line = re.sub(r'^\s*\?\?\?.*$', '', cleaned_line)
            cleaned_line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned_line) # Remove markdown links
            if cleaned_line.strip():
                current_section["text"] += cleaned_line + "\n"
                
    if current_section["title"] or current_section["text"] or current_section["code_blocks"]:
        sections.append(current_section)
        
    return sections

def generate_datasets(sections):
    datasets = []
    
    instructions = [
        "Answer the question about PyPDFForm.",
        "Provide guidance on using PyPDFForm.",
        "How do I use this PyPDFForm feature?",
        "Explain the following PyPDFForm functionality.",
        "PyPDFForm usage example request.",
        "Assistance with PyPDFForm.",
        "Explain the usage of PyPDFForm.",
        "Guidance on PyPDFForm implementation."
    ]

    for section in sections:
        title = section["title"]
        text = section["text"].strip()
        code_blocks = section["code_blocks"]
        
        if not code_blocks:
            continue

        # Adjust title for natural language questions
        verb_title = title
        if title.lower().startswith("handling "):
            verb_title = "handle " + title[9:]
        elif title.lower().startswith("using "):
            verb_title = "use " + title[6:]
        
        lower_title = verb_title.lower()
            
        for code in code_blocks:
            # Style 1: Direct usage question (3 variations)
            inputs_s1 = [
                f"How do I {lower_title}?",
                f"What is the way to {lower_title}?",
                f"Can you show me how to {lower_title} using PyPDFForm?"
            ]
            for inp in inputs_s1:
                datasets.append({
                    "instruction": random.choice(instructions),
                    "input": inp,
                    "output": f"{text[:200]}...\n\n```python\n{code}```" if len(text) > 200 else f"{text}\n\n```python\n{code}```"
                })
            
            # Style 2: Feature explanation (3 variations)
            inputs_s2 = [
                f"Explain {lower_title}.",
                f"Tell me about {title.lower()} in PyPDFForm.",
                f"How does {lower_title} work?"
            ]
            for inp in inputs_s2:
                datasets.append({
                    "instruction": f"Explain how to {lower_title} using PyPDFForm.",
                    "input": inp,
                    "output": f"{text}\n\n```python\n{code}```"
                })
            
            # Style 3: Troubleshooting style (3 variations)
            inputs_s3 = [
                f"I'm having trouble with {lower_title}. Can you show me an example?",
                f"Why is my {lower_title} not working? Any examples?",
                f"Help with {lower_title} please."
            ]
            for inp in inputs_s3:
                datasets.append({
                    "instruction": "Answer the question about PyPDFForm.",
                    "input": inp,
                    "output": f"Sure! To {lower_title}, you can use the following approach:\n\n```python\n{code}```"
                })
            
            # Style 4: Minimal answer (2 variations)
            inputs_s4 = [
                title,
                f"Example for {lower_title}"
            ]
            for inp in inputs_s4:
                datasets.append({
                    "instruction": "Provide a short usage example.",
                    "input": inp,
                    "output": f"```python\n{code}```"
                })
            
            # Style 5: Alternative question (2 variations)
            inputs_s5 = [
                f"Can you provide a code snippet for {lower_title}?",
                f"Snippet for {lower_title}."
            ]
            for inp in inputs_s5:
                datasets.append({
                    "instruction": random.choice(instructions),
                    "input": inp,
                    "output": f"Here is an example of {lower_title}:\n\n```python\n{code}```"
                })

            # Style 6: Detailed question (2 variations)
            inputs_s6 = [
                title,
                f"Tell me more about {lower_title}"
            ]
            for inp in inputs_s6:
                datasets.append({
                    "instruction": "Explain the following PyPDFForm feature.",
                    "input": inp,
                    "output": f"{text}\n\n```python\n{code}```"
                })

            # Style 7: "What's the syntax" (2 variations)
            inputs_s7 = [
                f"In PyPDFForm, what is the syntax for {lower_title}?",
                f"Syntax for {lower_title} in PyPDFForm."
            ]
            for inp in inputs_s7:
                datasets.append({
                    "instruction": "Provide the syntax for the following PyPDFForm operation.",
                    "input": inp,
                    "output": f"The syntax for {lower_title} is as follows:\n\n```python\n{code}```"
                })

            # Style 8: "Can I..." (2 variations)
            inputs_s8 = [
                f"Can I {lower_title} with PyPDFForm?",
                f"Is it possible to {lower_title} using this library?"
            ]
            for inp in inputs_s8:
                datasets.append({
                    "instruction": "Confirm if a feature is available and provide an example.",
                    "input": inp,
                    "output": f"Yes, PyPDFForm supports {lower_title}.\n\n```python\n{code}```"
                })

            # Style 9: "Walk me through" (2 variations)
            inputs_s9 = [
                f"Walk me through how to {lower_title}.",
                f"Step-by-step guide for {lower_title}."
            ]
            for inp in inputs_s9:
                datasets.append({
                    "instruction": "Provide a step-by-step example for PyPDFForm.",
                    "input": inp,
                    "output": f"To {lower_title}, follow these steps:\n\n```python\n{code}```"
                })

            # Style 10: "Simple example" (2 variations)
            inputs_s10 = [
                f"Give me a simple example of {lower_title}.",
                f"Basic usage of {lower_title}."
            ]
            for inp in inputs_s10:
                datasets.append({
                    "instruction": "Provide a basic PyPDFForm usage example.",
                    "input": inp,
                    "output": f"```python\n{code}```"
                })
            
    return datasets

def main():
    user_guide_files = [
        "docs/install.md",
        "docs/coordinate.md",
        "docs/font.md",
        "docs/prepare.md",
        "docs/inspect.md",
        "docs/fill.md",
        "docs/style.md",
        "docs/annotation.md",
        "docs/acro_js.md",
        "docs/draw.md",
        "docs/utils.md"
    ]
    
    all_sections = []
    for file_path in user_guide_files:
        if os.path.exists(file_path):
            all_sections.extend(extract_sections(file_path))
            
    datasets = generate_datasets(all_sections)
    
    print(f"Generated {len(datasets)} datasets.")
    
    with open("pypdfform_dataset.jsonl", "w") as f:
        for entry in datasets:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
