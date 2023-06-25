import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


from tqdm import tqdm

from unet import UNetModel
from ddpm import GaussianDiffusion

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightDataset(Dataset):
    def __init__(self, data_path, transform, test=False):
        self.data_path = data_path
        self.class_to_id = {"green": 0, "red": 1, "black": 2, "yellow": 3, "number": 4}
        self.image_paths = []
        self.labels = []
        self.crop_size = (32, 96)
        self.test = test
        self.transform = transform
        
        # Loop through each folder and gather image paths and labels
        for folder in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, folder)) and folder in self.class_to_id.keys():
                class_id = self.class_to_id[folder]
                folder_path = os.path.join(data_path, folder)
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(".jpg"):
                        image_path = os.path.join(folder_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(class_id)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        label = self.labels[index]
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        img = img.resize(self.crop_size, resample=Image.BILINEAR)
        img = self.transform(img)
        # img = ToTensor()(img)  # Convert the PIL image to a PyTorch tensor
        # img = img/255.0
        return img, label

batch_size = 64
timesteps = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
cwd = "/home/jinyujie/spirit/xiaoxiaojiang/ddpm"
dataset_path = os.path.join(cwd, "data", "SinLight")
dataset = LightDataset(dataset_path, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    channel_mult=(1, 2, 2, 2),
    attention_resolutions=(2,),
    dropout=0.1
)
model.to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

checkpoints_path = os.path.join(cwd, "checkpoints", "tlight")
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

