import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        
        # 1. Embedding Layer
        # Converts integer IDs (e.g., Word #45) into Vectors (Size 50)
        # We use a pre-made layer instead of loading GloVe manually for simplicity here
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer
        # batch_first=True means input format is (Batch, Sequence Length, Features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 3. Fully Connected Layer (Classifier)
        # Maps the final memory state to the output classes (Positive/Negative)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 4. Sigmoid (for probability 0-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch Size, Sentence Length] (e.g., 64 sentences, 20 words each)
        
        # 1. Embed
        # Output: [64, 20, 50] (Words become vectors)
        embedded = self.embedding(x)
        
        # 2. Pass through LSTM
        # output: Status of every step
        # (hidden, cell): The FINAL memory state after reading the whole sentence
        output, (hidden, cell) = self.lstm(embedded)
        
        # 3. We only care about the FINAL memory state (The summary of the sentence)
        # hidden[-1] is the last layer's hidden state
        final_memory = hidden[-1]
        
        # 4. Classify
        prediction = self.fc(final_memory)
        return self.sigmoid(prediction)

# Example Configuration
vocab_size = 1000  # We assume we only know 1000 unique words
embedding_dim = 50 # Vector size (like GloVe-50)
hidden_dim = 128   # Size of the LSTM's "Brain"/Memory
output_dim = 1     # Single number (0=Negative, 1=Positive)

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
print(model)

# Create a dummy batch of 3 sentences, each with 10 words
# The numbers represent Word IDs (indices in the vocabulary)
dummy_input = torch.randint(0, 1000, (3, 10)) 

output = model(dummy_input)

print(f"Input Shape: {dummy_input.shape} (3 Sentences, 10 Words)")
print(f"Output Shape: {output.shape} (3 Predictions)")
print(f"Predictions:\n{output.detach()}")