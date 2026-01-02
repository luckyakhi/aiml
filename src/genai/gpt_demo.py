from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Load Pre-trained GPT-2
# We use 'gpt2' (small version). There are larger ones like 'gpt2-medium'.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

print("GPT-2 Loaded!")

# 1. Encode Input
input_text = "The java developer decided to learn Python because"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 2. Generate
# max_length=50: Stop after 50 tokens
# num_return_sequences=1: Generate 1 story
output = model.generate(input_ids, max_length=20, num_return_sequences=1)

# 3. Decode (Numbers -> Words)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("--- Generated Story ---")
print(generated_text)

# set_seed ensures reproducibility for the demo
torch.manual_seed(42)

output_creative = model.generate(
    input_ids, 
    max_length=50, 
    num_return_sequences=1,
    do_sample=True,   # Enable randomness (Sampling)
    top_k=50,         # Limit to top 50 words
    temperature=0.7   # Medium creativity
)

print("--- Creative Story ---")
print(tokenizer.decode(output_creative[0], skip_special_tokens=True))