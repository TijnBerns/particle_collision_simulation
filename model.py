import torch.nn as nn
import torch
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, latent_dims, n_hidden):
        """
        Args:
            input_dims (int): Number of input dimensions.
            latent_dims (int): Number of latent dimensions.
            n_hidden (int): Number of nodes in the hidden layer.
        """
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dims, n_hidden)
        self.fc2 = nn.Linear(n_hidden, latent_dims)
        self.fc3 = nn.Linear(n_hidden, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        
    def forward(self, x): 
        """Forwards a sample through the network

        Args:
            x: Sample to be forwarded.

        Returns:
            z: Latent representation of input vector.
        """
        x = F.leaky_relu(self.fc1(x))
        mu = self.fc2(x)
        sigma = torch.exp(self.fc3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
        
        
class Decoder(nn.Module): 
    def __init__(self, latent_dims, output_dims, n_hidden) -> None:
        """
        Args:
            input_dims (int): Number of input dimensions.
            latent_dims (int): Number of latent dimensions.
            n_hidden (int): Number of nodes in the hidden layer.
        """
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dims, n_hidden)
        self.fc2 = nn.Linear(n_hidden, output_dims)
    
        
    def forward(self, x):
        """Constructs a vector based on the input 

        Args:
            x: Latent representation

        Returns:
            Predicted sample
        """
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)

        
        
class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dims, latent_dims, output_dims, n_hidden=400) -> None:
        """
        Args:
            input_dims (int): Number of input dimensions.
            latent_dims (int): Number of latent dimensions.
            output_dims (int): Number of dimensions of the output vector
            n_hidden (int): Number of nodes in the hidden layer.
        """
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VariationalEncoder(input_dims, latent_dims, n_hidden)
        self.decoder = Decoder(latent_dims, output_dims, n_hidden)
        
    def forward(self, x):
        """Forwards a smaple through the encoder and decoder

        Args:
            x: Sample to be forwarded

        Returns:
            Reconstructed sample
        """
        z = self.encoder(x)
        
        return self.decoder(z)
        
    
        
    