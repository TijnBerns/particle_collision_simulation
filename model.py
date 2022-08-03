from re import S
import torch.nn as nn
import torch
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, latent_dims, device="cpu"):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dims, input_dims * 2)
        self.fc2 = nn.Linear(input_dims * 2, input_dims * 3)
        self.fc3 = nn.Linear(input_dims * 3, latent_dims)
        self.fc4 = nn.Linear(input_dims * 3, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = torch.exp(self.fc4(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
        
        
class Decoder(nn.Module): 
    def __init__(self, latent_dims, output_dims) -> None:
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dims, latent_dims // 2)
        self.fc2 = nn.Linear(latent_dims // 2, latent_dims // 4)
        self.fc3 = nn.Linear(latent_dims // 4, output_dims)
    
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

        
        
class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        z = self.encoder(x)
        
        return self.decoder(z)
        
    
        
    