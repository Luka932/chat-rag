# src/generation.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Generator:
    def __init__(self, model_name='EleutherAI/gpt-j-6B'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt, temperature=0.7, max_length=150):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, temperature=temperature, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
