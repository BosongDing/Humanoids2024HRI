import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle


np.random.seed(0)
traj_length = 60
z_length = 10
hidden_dim = 64
hidden_dim1 = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load your dataset
data =pickle.load(open('traj_raw.pkl','rb'))
print(data[0].shape)#(34,2)

#pad the traj with -999 to 60 length in the end
for i in range(len(data)):
    data[i] = np.concatenate((data[i], 0*np.ones((traj_length-len(data[i]), 2))), axis=0)
data = np.array(data)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, z_length)  # Assuming `mus` has 10 features as in your VAE's latent dimension
        )

    def forward(self, x):
        return self.layers(x)

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

model = VAE()
model.load_state_dict(torch.load('model_new_60_cumsum.pth', map_location=device))
model.eval()
model = model.to(device)


original_trajectories = data[:]
reconstructed_trajectories, mus, log_stds = model(torch.Tensor(original_trajectories).view(-1, traj_length*2).to(device))
reconstructed_trajectories = reconstructed_trajectories.view(-1, traj_length, 2).detach().cpu().numpy()
mus = mus.detach().cpu().numpy()


label_final_yaw = []
label_final_pitch = []
for i in range(len(original_trajectories)):
    label_final_yaw.append( np.sum(original_trajectories[i,:,0]) )
    label_final_pitch.append( np.sum(original_trajectories[i,:,1]) )

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#latents_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(mus)
latents_embedded = PCA(n_components=2).fit_transform(mus)

latents_embedded = mus


label_final_yaw = np.array(label_final_yaw)
label_final_pitch = np.array(label_final_pitch)
print(label_final_yaw.min(), label_final_yaw.max(), label_final_pitch.min(), label_final_pitch.max())

ymax = 35
pmax = 25
label_final_yaw /= ymax
label_final_pitch /= pmax

color_yaw = []
color_pitch = []
final_yaw_norm = []
final_pitch_norm = []
for i in range(len(label_final_yaw)):
    final_yaw_norm.append( label_final_yaw[i] )
    final_pitch_norm.append( label_final_pitch[i] )
    color_yaw.append( plt.get_cmap('cool')( (final_yaw_norm[-1]+1.0)/2.0) )
    color_pitch.append( plt.get_cmap('cool')( (final_pitch_norm[-1]+1.0)/2.0) )

plt.subplot(1,2,1)
plt.scatter(latents_embedded[:,0], latents_embedded[:,1], color=color_yaw)
plt.subplot(1,2,2)
plt.scatter(latents_embedded[:,0], latents_embedded[:,1], color=color_pitch)
plt.show()



xdata = []
for i in range(len(final_yaw_norm)):
    xdata.append( [final_yaw_norm[i], final_pitch_norm[i]] )

net = MLP()
net.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
xdata_tensor = torch.tensor(xdata, dtype=torch.float32).to(device)
mus_tensor = torch.tensor(mus, dtype=torch.float32).to(device)
epochs = 10000
tolerance = 1e-4
last_loss = np.inf

for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()
    outputs = net(xdata_tensor)
    loss = criterion(outputs, mus_tensor)
    loss.backward()
    optimizer.step()

    # Simple stopping criteria based on loss improvement
    if np.abs(last_loss - loss.item()) < tolerance :
        print(f"Stopping training at epoch {epoch}, loss improvement is less than {tolerance}")
        torch.save(net.state_dict(), 'net_new_60_cumsum.pth')
        break
    last_loss = loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}","===================")

"""
for val in [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]:
    mu = net.predict( [[val,-0.5]] )
    print(mu)
    mu = torch.Tensor(mu).to(device)
    out_traj = model.decoder(mu).view(-1, traj_length, 2).detach().cpu().numpy()[0]

    plt.plot(np.cumsum(out_traj[:, 0]), np.cumsum(out_traj[:, 1]), color='orange')
plt.xlim([-20,20])
plt.ylim([-20,20])
plt.show()
#"""

# For each point on a 2D grid, predict the cumsum end
lims = 1.0
trajs = []
for yaw in np.linspace(-lims,lims,20):
    for pitch in np.linspace(-lims,lims,20):
        mu =net(torch.Tensor([[yaw,pitch]]).to(device))      
        # mu = torch.Tensor(mu).to(device)
        out_traj = model.decoder(mu).view(-1, traj_length, 2).detach().cpu().numpy()[0]
        # np.save("./traj"+str(yaw)+str(pitch)+".npy",out_traj)
        trajs.append(out_traj)
        plt.arrow(yaw, pitch, np.sum(out_traj[:, 0])/ymax-yaw, np.sum(out_traj[:, 1])/pmax-pitch, head_width=0.005, alpha=0.1)

        plt.plot([yaw], [pitch], '.k', alpha=0.1)
        plt.plot([np.sum(out_traj[:, 0])/ymax], [np.sum(out_traj[:, 1])/pmax], '.k')
        
       

colors = ['red', 'blue']  # Colors to alternate between
fig, ax = plt.subplots()
for trajectory in trajs:
    cumsum_x = np.cumsum(trajectory[:,0])
    cumsum_y = np.cumsum(trajectory[:,1])
    
    for i in range(len(cumsum_x) - 1):
        # Alternating colors for each segment
        color = colors[i % 2]
        # Plotting line segment between current and next point
        ax.plot(cumsum_x[i:i+2], cumsum_y[i:i+2], color=color)

plt.show()
plt.show()


"""
lims = 1
color_yaw = []
color_pitch = []
mus = []
for yaw in np.linspace(-lims,lims,20):
    for pitch in np.linspace(-lims,lims,20):
        mu = net.predict( [[yaw,pitch]] )
        mu = torch.Tensor(mu).to(device)
        out_traj = model.decoder(mu).view(-1, traj_length, 2).detach().cpu().numpy()[0]

        color_yaw.append( plt.get_cmap('cool')( (yaw+1.0)/2.0) )
        color_pitch.append( plt.get_cmap('cool')( (pitch+1.0)/2.0) )
        mus.append(mu.cpu().detach().numpy())

mus = np.concatenate(mus)
plt.subplot(1,2,1)
plt.scatter(mus[:,0], mus[:,1], color=color_yaw)
plt.subplot(1,2,2)
plt.scatter(mus[:,0], mus[:,1], color=color_pitch)
plt.show()
"""
