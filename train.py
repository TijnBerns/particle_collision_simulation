import torch
import model
from torch.utils.data import DataLoader
import data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import utils


def elbo_loss(x, x_hat, vae):
    """Implementation of the ELBO-loss function

    Args:
        x: input vector of vae network
        x_hat: output vector of vae network 
        vae: the vae network

    Returns:
        float: elbo loss value
    """
    return 10 * ((x - x_hat)**2).sum() + vae.encoder.kl


def train_epoch(vae, dataloader, optimizer, device='cpu'):
    """Trains the vae network for one epoch

    Args:
        vae: The vae network
        dataloader: The dataloader to train on
        optimizer: The optimizer
        device (str, optional): The device that is used

    Returns:
        The mean loss over the entire epoch
    """
    vae.train()
    train_loss = 0.0

    for x in dataloader:
        x = x.float()
        x = x.to(device)

        x_hat = vae(x)
        loss = elbo_loss(x, x_hat, vae)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(dataloader.dataset)


def test_epoch(vae, dataloader):
    """Tests the vae network

    Args:
        vae: The vae network
        dataloader: The dataloader to train on

    Returns:
        The mean loss over the entire dataset
    """
    vae.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x in dataloader:
            x = x.float()
            z = vae.encoder(x)
            x_hat = vae.decoder(z)
            loss = elbo_loss(x, x_hat, vae)
            test_loss += loss.item()

    return test_loss / len(dataloader.dataset)


def train(vae, train_loader, valid_loader, epochs, latent_dims):
    """Train the vae network on the train loader for the specified number of epochs

    Args:
        vae (_type_): the vae network that is trained
        train_loader (_type_): the loader that is used to perform the training on
        valid_loader (_type_): the loader that is used to validate the network performance
        epochs (_type_): the number of epochs the network is trained
        latent_dims (int, optional): the number of latent dimensions 

    Returns:
        float: the minium loss of on the validation loader
    """
    # Initiliaze optimizer
    lr = 1e-3
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

    # Train the network 
    min_loss = 1_000_000
    loss = []
    for i in range(epochs):
        train_loss = train_epoch(vae, train_loader, optim)
        test_loss = test_epoch(vae, valid_loader)

        print(
            f"epoch: {i:<3} train_loss: {train_loss:<10.5f} test_loss: {test_loss:<10.5f}")
        loss.append([train_loss, test_loss])

        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(vae.decoder.state_dict(),
                       f'models/decoder_{latent_dims}.pt')

    loss = pd.DataFrame(loss, columns=['train_loss', 'test_loss'])
    loss.to_csv(f'results/loss_{latent_dims}.csv')
    return min_loss


def generate_data(decoder, scaler, input_dims, latent_dims, n=100_000, device='cpu'):
    """Generates data given a decoder network

    Args:
        decoder: The decoder network that is used
        scaler: The scaler that is used during training of the decoder
        input_dims: The number of dimensions of the input vector
        latent_dims: The number of dimensions of the latent representation
        n (optional): The number of samples that are generated. Defaults to 100_000.
        device (str, optional): The device that is used. Defaults to 'cpu'.
    """
    decoder.eval()
    with torch.no_grad():
        latent = torch.randn(n, latent_dims, device=device)
        data = decoder(latent).reshape(n, input_dims)
        data = scaler.inverse_transform(data)

    # Construct dataframe and save to csv
    df = pd.DataFrame(
        data, columns=["E1", "M1", "p_x1", "p_y1", "p_z1", "E2", "M2", "p_x2", "p_y2", "p_z2"])
    df.to_csv(f"data/vae.csv")


if __name__ == "__main__":
    # Set the random seed and the device
    torch.manual_seed(2022)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device:<20}")

    # Training parameters
    input_dims = 10
    latent_dims = 3
    n_hidden = 384
    epochs = 50
    batch_size = 32

    # Initialize the train and test sets
    dataset = data.HEPDataset("data/mc_sim.csv")
    train_set, valid_set = data.split(dataset)

    # Scale the data
    print("Scaling the training data...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_set = scaler.fit_transform(train_set)
    valid_set = scaler.transform(valid_set)

    # Initialize dataloaders
    train_loader = DataLoader(train_set, batch_size=32,
                              drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32,
                              drop_last=True, shuffle=False)

    # Initialize the netwrok and and train it
    print("Training the network...")
    vae = model.VariationalAutoEncoder(
        input_dims, latent_dims, input_dims, n_hidden)
    train(vae, train_loader, valid_loader,
          latent_dims=latent_dims, epochs=epochs, device=device)

    # Initialize the decoder
    decoder = model.Decoder(latent_dims, input_dims, n_hidden)
    decoder.load_state_dict(torch.load(f"models/decoder_{3}.pt"))

    # Generate data
    generate_data(decoder, scaler, input_dims, latent_dims)

    # Plot results
    mc_data = pd.read_csv('data/mc_sim_full.csv', dtype=float, header=0)
    vae_data = pd.read_csv('data/vae.csv', dtype=float, header=0)
    utils.plot_all(mc_data, vae_data)
