# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load a model designed for Semantic Search
# 'all-MiniLM-L6-v2' is fast and effective
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedder Loaded!")

# Our "Database" of internal documents
knowledge_base = [
    "Policy A: Transaction passwords must be reset every 90 days via the secure portal.",
    "Policy B: International wire transfers above $50k require Level 2 approval from a VP.",
    "Policy C: Employees can claim up to $500 for home office equipment every 2 years.",
    "Policy D: The cafeteria serves Sushi only on Fridays."
]

# 2. Convert Knowledge Base to Vectors (Index the data)
kb_embeddings = embedder.encode(knowledge_base, convert_to_tensor=True)

print("Knowledge Base Indexed!")

def retrieve_context(user_query):
    # 1. Embed the User Query
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    
    # 2. Search (Cosine Similarity)
    # Compare query vs. all 4 documents
    scores = util.cos_sim(query_embedding, kb_embeddings)[0]
    
    # 3. Get the best match (Top 1)
    best_score_idx = torch.argmax(scores).item()
    return knowledge_base[best_score_idx]

# Test it
query = "How do I approve a large transfer to London?"
context = retrieve_context(query)

print(f"User Query: {query}")
print(f"Retrieved Context: {context}")

def generate_rag_prompt(query):
    # 1. Retrieve
    context = retrieve_context(query)
    
    # 2. Augment
    prompt = f"""
    You are a helpful banking assistant. Answer the user's question using ONLY the context provided below.
    
    Context: {context}
    
    Question: {query}
    """
    return prompt

print("--- Final Prompt to send to LLM ---")
print(generate_rag_prompt("Can I expense a new monitor?"))