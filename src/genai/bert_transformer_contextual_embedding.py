# Install the library (if running locally, in Colab it's usually pre-installed)
# !pip install transformers

from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 1. Load Pre-trained BERT
# 'bert-base-uncased' is the standard version (lower case only)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

print("BERT Model Loaded!")

# Helper function to get the vector for a specific word in a sentence
def get_word_vector(sentence, word_index):
    # 1. Tokenize (Convert string to IDs)
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # 2. Run BERT (Forward pass)
    # output.last_hidden_state shape: [1, Sentence_Length, 768] (768 dimensions per word)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Extract the vector for the target word
    # inputs.input_ids[0] gives us the list of token IDs. We need to find where the word is.
    # For simplicity, I'm manually passing the index of the word "bank" 
    # (Note: BERT adds [CLS] at start, so index is word_pos + 1)
    word_vector = outputs.last_hidden_state[0][word_index]
    return word_vector

# Sentence 1: "I deposited money at the bank" 
# bank is the 6th word (index 6 because of [CLS])
vec_financial = get_word_vector("I deposited money at the bank", 6)

# Sentence 2: "I sat on the river bank"
# bank is the 6th word (index 6)
vec_nature = get_word_vector("I sat on the river bank", 6)

# Calculate Similarity (Cosine Similarity)
cos = torch.nn.CosineSimilarity(dim=0)
similarity = cos(vec_financial, vec_nature)

print(f"Similarity between 'Bank' (Money) and 'Bank' (River): {similarity.item():.4f}")