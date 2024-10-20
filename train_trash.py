import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, get_embedding, tokenize, stem
from model import NeuralNet

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    # Add to tag list
    tags.append(tag)
    for pattern in intent["patterns"]:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        # Add to our words list
        all_words.extend(w)
        # Add to xy pair
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Hyper-parameters
num_epochs = 2000
batch_size = 16
learning_rate = 0.001
input_size = 300  # Dimensionality of input features
hidden_size = 32
output_size = len(tags)
max_length = 300  # Maximum length for padding

print(input_size, output_size)

# Create training data
X_train = []
y_train = []

for pattern_sentence, tag in xy:
    label = tags.index(tag)
    y_train.append(label)  # Append the label

    # Get embeddings for each word in the pattern sentence
    embeddings = [get_embedding(w) for w in pattern_sentence]

    # Ensure all embeddings have the same length
    padded_embeddings = []
    for emb in embeddings:
        if len(emb) < max_length:
            # Pad the embedding
            padded_emb = np.pad(emb, (0, max_length - len(emb)), "constant")
        else:
            # Truncate the embedding
            padded_emb = emb[:max_length]
        padded_embeddings.append(padded_emb)

    # Convert padded embeddings to a NumPy array
    X_train.append(np.array(padded_embeddings))

# Convert to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Print sizes to verify
print("Size of X_train:", X_train.shape)
print("Size of y_train:", y_train.shape)


# Custom Dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Create DataLoader
dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.float().to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

# Save the model data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data1.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
