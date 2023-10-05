from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchsummary
import torch.jit


class DCGANGenerator (nn.Module):
    def __init__(self, kNoiseSize: int):
        super(DCGANGenerator, self).__init__();
        # layer 1
        self.conv1 = nn.ConvTranspose2d(kNoiseSize, 256, 4, bias=False);
        self.batch_norm1 = nn.BatchNorm2d(256);
        # layer 2
        self.conv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, bias=False);
        self.batch_norm2 = nn.BatchNorm2d(128);
        # layer 3
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False);
        self.batch_norm3 = nn.BatchNorm2d(64);
        # layer 4 (produce output)
        self.conv4 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False);
        
        # layer 1
        self.register_module("conv1", self.conv1);
        self.register_module("batch_norm1", self.batch_norm1);
        # layer 2
        self.register_module("conv2", self.conv2);
        self.register_module("batch_norm2", self.batch_norm2);
        # layer 3
        self.register_module("conv3", self.conv3);
        self.register_module("batch_norm3", self.batch_norm3);
        # layer 4 (produce output)
        self.register_module("conv4", self.conv4);
    
    def forward(self, x: torch.Tensor):
        # layer 1
        x = torch.relu(self.batch_norm1(self.conv1(x)));
        # layer 2
        x = torch.relu(self.batch_norm2(self.conv2(x)));
        # layer 3
        x = torch.relu(self.batch_norm3(self.conv3(x)));
        # layer 4 (produce output)
        x = torch.tanh(self.conv4(x));
        return x;
    
    
discriminator = nn.Sequential(
    # layer 1
    nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(negative_slope=0.2),
    # layer 2
    nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(negative_slope=0.2),
    # layer 3
    nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(negative_slope=0.2),
    # layer 4
    nn.Conv2d(256, 1, 3, stride=1, padding=0, bias=False),
    nn.Sigmoid()
);


if (__name__ == "__main__"):
    device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Initialize device", end=" :: ")
    print(device)
    
    print("Initialize generator")
    kNoiseSize: int = 100
    generator: DCGANGenerator = DCGANGenerator(kNoiseSize)
    generator.eval()
    # generator.to(device)
    
    print("Initialize discriminator")
    discriminator.eval()
    # discriminator.to(device)
    
    print("Combine DCGAN")
    dcgan = nn.Sequential(generator, discriminator)
    
    model_file = "jit_script_dcgan.pt"
    torch_script = torch.jit.script(dcgan)
    torch_script.save(model_file)
    print("Save JIT script model.. ::", model_file, "::", type(torch_script))    
    torch_script = torch.jit.load(model_file)
    print("Load JIT script model.. ::", model_file, "::", type(torch_script))
    
    # summary model
    print("\nDCGAN:")
    torchsummary.summary(model=dcgan, input_data=[100, 1, 1]);
    print("\nJIT DCGAN:")
    torchsummary.summary(model=torch_script, input_data=[100, 1, 1]);
