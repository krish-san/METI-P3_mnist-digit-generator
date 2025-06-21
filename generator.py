import torch
import torch.nn as nn

latent_dim = 100
num_classes = 10

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_embed(labels)
        x = torch.cat([z, label_input], dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)
