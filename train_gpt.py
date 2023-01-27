# %%
from transformers import GPT2Tokenizer, GPT2Model
import torch as t

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
text = "Replace me by any text you'd like."
eos_token = "<|endoftext|>"
prompt = text
print("Starting prompt: " + prompt)
# %%
for i in range(100):
    encoded_input = tokenizer(prompt, return_tensors="pt")
    output = model(**encoded_input)
    mlp_output = output.last_hidden_state[0, -1, :]
    cos_similarity = t.nn.functional.cosine_similarity(mlp_output, model.wte.weight, dim=1)
    next_token = cos_similarity.argmax()
    next_string = tokenizer.decode(next_token)
    if next_string == eos_token:
        break
    prompt = prompt + next_string
    print("Current generation: " + prompt)

# %%
