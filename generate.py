import torch
import monte_carlo
import train
import data
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
import model


if __name__ == "__main__":
    path = "mc"
    if not Path('data/mc_sim.csv').is_file():
        os.makedirs('data')
        monte_carlo.monte_carlo_sim()
        
    # Network parameters
    input_dims = 10
    output_dims = 10
    latent_dims = 3
    n_hidden = 384
    epochs = 50
    batch_size = 32
    
    # Initialize the train and test sets
    dataset = data.HEPDataset("data/mc_sim.csv")
    train_set, _ = data.split(dataset)

    # Scale the data
    print("Fitting scaler to data...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_set = scaler.fit_transform(train_set)
    
    # Generate data using VAE
    print("Generating data using VAE...")
    decoder = model.Decoder(latent_dims, output_dims, n_hidden)
    decoder.load_state_dict(torch.load('models/decoder_3.pt'))
    train.generate_data(decoder, scaler, input_dims, latent_dims)
        
    

    
