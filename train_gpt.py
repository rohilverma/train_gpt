# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch as t

def test_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    text = "Replace me by any text you'd like."
    eos_token = "<|endoftext|>"
    prompt = text
    print("Starting prompt: " + prompt)

    for i in range(100):
        encoded_input = tokenizer(prompt, return_tensors="pt")
        next_token = model(**encoded_input).logits[0, -1].argmax()
        next_string = tokenizer.decode(next_token)
        if next_string == eos_token:
            break
        prompt = prompt + next_string
        print("Current generation: " + prompt)

# %%

class GPT2Block(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        

    def forward(self, x):
        pass

class GPT2Clone(t.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.wte = t.nn.Embedding(50257, 768)
        self.wpe = t.nn.Embedding(1024, 768)
        self.drop = t.nn.Dropout(0.1)
    
    def forward(self, x):

    

