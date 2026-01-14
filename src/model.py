import torch
import torch.nn as nn

class SensitiveDataClassifier(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        return self.fc(x)
