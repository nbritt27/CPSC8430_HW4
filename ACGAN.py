import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.utils import save_image
output_dir = 'ACGAN'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

c_losses=[]
g_losses=[]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.noise_input = nn.Linear(100, 4*4*512)
        self.text_input = nn.Linear(119, 256)
        self.relu = nn.ReLU()
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(512, 0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(256, 0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(128, 0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(64, 0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 5, 2, 2, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, text):
        n_out = self.noise_input(noise)
        n_out = n_out.view(-1, 512, 4, 4)
        
        t_out = self.relu(self.text_input(text))
        
        combined = torch.cat((n_out, t_out.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4,4)), dim=1)[:, :512]
        
        return self.model(combined)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.image_input = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128, 0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256, 0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512, 0.9),
            nn.LeakyReLU(0.2)
        )
        
        self.text_input = nn.Sequential(
            nn.Linear(119, 256),
            nn.ReLU()
        )
        
        self.model = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1, 0),
            nn.Flatten(),
            nn.Linear(512*4*4, 1)
        )
        
        self.aux_model = nn.Sequential(
            nn.Linear(512*4*4, 119),
            nn.Sigmoid()
        )
        
    def forward(self, image, text):
        i_out = self.image_input(image)
        
        t_out = self.text_input(text)
        t_out = t_out.view(-1, 256, 1, 1).repeat(1, 1, 4, 4)
        
        combined = torch.cat((i_out, t_out), dim=1)
        
        validity = self.model(combined)
        label = self.aux_model(combined.view(combined.size(0), -1))

        return validity, label


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

G = Generator().cuda()
D = Discriminator().cuda()

adversarial_loss = nn.MSELoss()
auxiliary_loss = nn.CrossEntropyLoss()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 100

for epoch in range(epochs):
    for i, (images, labels) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.cuda()
        labels = labels.cuda()
        
        D.zero_grad()
        text = torch.randn(batch_size, 119).cuda()
        
        real_validity, real_labels = D(images, text)
        d_real_loss = (adversarial_loss(real_validity, torch.ones_like(real_validity)) + 
                       auxiliary_loss(real_labels, labels)) / 2

        noise = torch.randn(batch_size, 100).cuda()
        fake_images = G(noise, text)
        fake_validity, fake_labels = D(fake_images.detach(), text)
        d_fake_loss = (adversarial_loss(fake_validity, torch.zeros_like(fake_validity)) + 
                       auxiliary_loss(fake_labels, labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_losses.append(d_loss)
        
        d_loss.backward()
        D_optimizer.step()
        
        G.zero_grad()
        noise = torch.randn(batch_size, 100).cuda()
        fake_images = G(noise, text)
        fake_validity, fake_labels = D(fake_images, text)
        g_loss = (adversarial_loss(fake_validity, torch.ones_like(fake_validity)) + 
                  auxiliary_loss(fake_labels, labels)) / 2
        
        g_losses.append(g_loss)
        g_loss.backward()
        G_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    with torch.no_grad():
        fixed_noise = torch.randn(batch_size, 100).cuda()
        fixed_text = torch.randn(batch_size, 119).cuda()
        generated_images = G(fixed_noise, fixed_text)
        save_image(generated_images, os.path.join(output_dir, f'generated_images_epoch_{epoch+1}.png'), normalize=True)