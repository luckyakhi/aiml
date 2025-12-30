import gensim.downloader as api

# Download a smaller version of the GloVe model (trained on Wikipedia)
# This model knows 50 numbers (dimensions) for 400,000 words.
print("Downloading model... (This may take a moment)")
word_vectors = api.load("glove-wiki-gigaword-50")
print("Model Loaded!")

# 1. Check Similarity
# How close are 'apple' and 'banana'?
similarity = word_vectors.similarity('apple', 'banana')
print(f"Similarity(Apple, Banana): {similarity:.2f}")

# How close are 'apple' and 'car'?
similarity = word_vectors.similarity('apple', 'phone')
print(f"Similarity(Apple, Phone):    {similarity:.2f}")

# 2. The Analogy Test
# King - Man + Woman = ?
result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"\nKing - Man + Woman = {result[0][0]} (Confidence: {result[0][1]:.2f})")

# 3. Try your own! (e.g., Paris - France + Italy = ?)
result2 = word_vectors.most_similar(positive=['italy', 'paris'], negative=['france'], topn=1)
print(f"Paris - France + Italy = {result2[0][0]}")