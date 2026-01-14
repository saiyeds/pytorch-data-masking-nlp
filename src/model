
import torch
import torch.nn as nn
from model import SensitiveDataClassifier

def train_classifier():
    model = SensitiveDataClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model
