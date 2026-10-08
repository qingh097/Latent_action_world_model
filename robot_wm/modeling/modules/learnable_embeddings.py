import torch
import torch.nn as nn


class LearnableEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init="xavier"):
        """
        num_embeddings: number of embeddings
        embedding_dim: dimension of each embedding vector
        init: initialization method ('xavier', 'normal', 'uniform', etc.)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Define learnable parameters
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        # Initialize embeddings
        if init == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif init == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        elif init == "uniform":
            nn.init.uniform_(self.weight, a=-0.05, b=0.05)
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, indices=None):
        """
        indices: optional LongTensor of indices to select embeddings
                 If None, return all embeddings.
        """
        if indices is None:
            return self.weight
        return self.weight[indices]
