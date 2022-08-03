from sched import scheduler
import torch
import model
from torch.utils.data import DataLoader
import data
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def vae_loss(x, x_hat, vae):
    return 100 * ((x - x_hat)**2).sum() + vae.encoder.kl


def train_epoch(vae, dataloader, optimizer, device='cpu'):
    vae.train()
    train_loss = 0.0

    for x in tqdm(dataloader):
        x = x.float()
        x = x.to(device)

        x_hat = vae(x)
        loss = vae_loss(x, x_hat, vae)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(dataloader.dataset)


def test_epoch(vae, dataloader):
    vae.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x in dataloader:
            x = x.float()
            z = vae.encoder(x)
            x_hat = vae.decoder(z)
            loss = vae_loss(x, x_hat, vae)
            test_loss += loss.item()

    return test_loss / len(dataloader.dataset)


def train(train_loader, valid_loader, epochs, input_dims, latent_dims, device="cpu"):
    # Initialize the network
    encoder = model.VariationalEncoder(input_dims, latent_dims)
    decoder = model.Decoder(latent_dims, input_dims)
    vae = model.VariationalAutoEncoder(encoder, decoder).to(device)

    # Initiliaze optimizer
    lr = 1e-3
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    # lr = 1e-4
    # optim = torch.optim.SGD(vae.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, epochs, eta_min=0)

    # Train the network
    min_loss = 1_000_000
    for i in range(epochs):
        train_loss = train_epoch(vae, train_loader, optim)
        test_loss = test_epoch(vae, valid_loader)
        # scheduler.step()

        print(
            f"epoch: {i:<3} train_loss: {train_loss:<10.5f} test_loss: {test_loss:<10.5f}")

    torch.save(vae.decoder.state_dict(), 'models/decoder.pt')


def generate_data(decoder, scaler, input_dims, latent_dims, n=100_000, device='cpu'):
    decoder.eval()
    data = []
    with torch.no_grad():

        latent = torch.randn(n, latent_dims, device=device)
        event = decoder(latent).reshape(n, input_dims)
        event = scaler.inverse_transform(event)

    data = pd.DataFrame(
        event, columns=["E1", "p_x1", "p_y1", "p_z1", "E2", "p_x2", "p_y2", "p_z2"])

    data.to_csv(f"data/vae.csv")


if __name__ == "__main__":

    torch.manual_seed(2022)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device:<20}")

    # Training parameters
    input_dims = 8
    latent_dims = 4 * input_dims
    epochs = 40
    batch_size = 16

    # Initialize the train and test sets
    dataset = data.HEPDataset("data/mc_sim.csv")
    train_set, valid_set = data.split(dataset)

    # Scale the data
    print("Scaling the training data")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_set = scaler.fit_transform(train_set)
    # test_set = scaler.transform(test_set)
    valid_set = scaler.transform(valid_set)

    # Initialize dataloaders
    train_loader = DataLoader(train_set, batch_size=32,
                              drop_last=True, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=32,
    #                          drop_last=True, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=32,
                              drop_last=True, shuffle=False)

    # Train the network
    print("Training the network")
    train(train_loader, valid_loader, epochs=epochs, input_dims=input_dims,
          latent_dims=latent_dims, device=device)

    # Initialize the decoder
    decoder = model.Decoder(latent_dims, input_dims)
    decoder.load_state_dict(torch.load("models/decoder.pt"))

    # Generate data
    generate_data(decoder, scaler, input_dims, latent_dims)
