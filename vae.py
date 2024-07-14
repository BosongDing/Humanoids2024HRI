import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
np.random.seed(0)
traj_length = 60
z_length = 5
hidden_dim = 64
hidden_dim1 = 128
epochs = 200
KL_weight = 0.001 #0.005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load your dataset
data =pickle.load(open('traj_raw.pkl','rb'))
print(data[0].shape)#(34,2)

#pad the traj with -999 to 60 length in the end
for i in range(len(data)):
    data[i] = np.concatenate((data[i], 0*np.ones((traj_length-len(data[i]), 2))), axis=0)
data = np.array(data)
data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

# Create a dataset and data loader
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(traj_length*2, hidden_dim1),  
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_length*2),  
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, traj_length*2),
        )
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = x.view(-1, traj_length*2)  # Flatten the input
        encoded = self.encoder(x)
        mu, log_var = encoded.split(z_length, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()

# Training loop
min_loss = float('inf')
for epoch in range(epochs):
    total_recon = 0
    total_kl = 0
    for batch in dataloader:
        optimizer.zero_grad()
        x = batch[0].view(-1, traj_length*2)  # Flatten the input
        reconstructed, mu, log_var = model(x)
        # Compute the loss
        recon_loss = loss_function(reconstructed, x)
        kl_div = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        loss = recon_loss + KL_weight*kl_div

        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()

    if loss.item() <= min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), 'model_new_60.pth')

    print("epoch:"+ str(epoch),loss.item(), '\t\t', total_recon, '\t', total_kl)
        
import matplotlib.pyplot as plt

# Function to generate new trajectories
def generate_new_trajectories(model, num_samples=5):
    with torch.no_grad():
        z = torch.randn(num_samples, z_length).to(device)  # Assuming the latent space has dimension 10
        generated_trajectories = model.decoder(z).view(num_samples, traj_length, 2)  # Reshape to match the original data's shape
    return generated_trajectories.cpu().numpy()

# Generate new examples
new_trajectories = generate_new_trajectories(model, num_samples=8)

# Select a few original trajectories for comparison
original_trajectories = data[:]
reconstructed_trajectories, mus, log_stds = model(torch.Tensor(original_trajectories).view(-1, traj_length*2).to(device))
reconstructed_trajectories = reconstructed_trajectories.view(-1, traj_length, 2).detach().cpu().numpy()

# Plotting the comparisons
fig, axs = plt.subplots(8, 3, figsize=(10, 20))
for i in range(4):
    # Original Trajectories
    axs[i, 0].plot(np.cumsum(original_trajectories[i*15][:, 0]), np.cumsum(original_trajectories[i*15][:, 1]))
    axs[i, 0].set_title(f'Original Trajectory {i*15+1}')

    # Reconstructed Trajectories
    axs[i, 1].plot(np.cumsum(reconstructed_trajectories[i*15][:, 0]), np.cumsum(reconstructed_trajectories[i*15][:, 1]))
    axs[i, 1].set_title(f'Reconstructed Trajectory {i*15+1}')

    # Generated Trajectories
    axs[i, 2].plot(np.cumsum(new_trajectories[i][:, 0]), np.cumsum(new_trajectories[i][:, 1]), color='orange')
    axs[i, 2].set_title(f'Generated Trajectory {i+1}')

    axs[i,0].set_xlim([-10,10])
    axs[i,0].set_ylim([-10,10])
    axs[i,1].set_xlim([-10,10])
    axs[i,1].set_ylim([-10,10])
    axs[i,2].set_xlim([-10,10])
    axs[i,2].set_ylim([-10,10])

plt.tight_layout()
plt.show()



new_trajectories = generate_new_trajectories(model, num_samples=100)

fig, ax = plt.subplots()
for i in range(len(new_trajectories)):
    ax.plot(np.cumsum(new_trajectories[i][:,0]), np.cumsum(new_trajectories[i][:,1]))
plt.show()


