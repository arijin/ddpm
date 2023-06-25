import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm

from unet import UNetModel
from ddpm import GaussianDiffusion

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
timesteps = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
cwd = os.getcwd()
dataset_path = os.path.join(cwd, "data")
os.makedirs(dataset_path, exist_ok=True)
dataset = datasets.MNIST(dataset_path, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)
model.to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

checkpoints_path = os.path.join(cwd, "checkpoints", "mnist")
os.makedirs(checkpoints_path, exist_ok=True)

# train
epochs = 10

for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_size = images.shape[0]
        images = images.to(device)
        
        # sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        loss = gaussian_diffusion.train_losses(model, images, t)
        
        if step % 200 == 0:
            print("Loss:", loss.item())
            
        loss.backward()
        optimizer.step()  
    
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoints_path, f"model_{epoch}.pth"))
        # torch.save(optimizer.state_dict(), os.path.join(checkpoints_path, f"optimizer_{epoch}.pth"))
torch.save(model.state_dict(), os.path.join(checkpoints_path, f"model_{epoch}.pth"))  

