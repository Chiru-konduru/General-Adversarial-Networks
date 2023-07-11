import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
import os
sys.path.append(os.path.abspath("/assignment4_materials/gan"))
from gan.SpectralNorm import SpectralNorm


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
#         self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels=128, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size =4, stride = 2, padding = 1)
#         self.bn2 = nn.BatchNorm2d(512)
#         self.conv4 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size= 4, stride = 2, padding = 1)
#         self.bn3 = nn.BatchNorm2d(1024)
#         self.conv5 = nn.Conv2d(in_channels = 1024, out_channels = 1, kernel_size= 4, stride = 1, padding = 1)
#         self.leaky_Relu = nn.LeakyReLU(0.2)

        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, kernel_size = 4, stride = 2, padding = 1))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1))
        self.bn1 = nn.BatchNorm2d(256)            
        self.conv3 = SpectralNorm(nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1))
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = SpectralNorm(nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1))
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = SpectralNorm(nn.Conv2d(1024, 1, kernel_size = 4, stride = 1, padding = 1))
        self.leaky_Relu = nn.LeakyReLU(0.2)
                               
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        x = self.leaky_Relu(self.conv1(x))
        x = self.leaky_Relu(self.conv2(x))
        x = self.leaky_Relu(self.bn1(x))
        x = self.leaky_Relu(self.conv3(x))
        x = self.leaky_Relu(self.bn2(x))
        x = self.leaky_Relu(self.conv4(x))
        x = self.leaky_Relu(self.bn3(x))
        x = self.leaky_Relu(self.conv5(x))
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = nn.ConvTranspose2d(in_channels = self.noise_dim, out_channels = 1024, kernel_size = 4, stride = 1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(in_channels = 128, out_channels = 3, kernel_size = 4, stride = 2, padding = 1)
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn4(x))
        x = torch.tanh(self.conv5(x))  
        
        ##########       END      ##########
        
        return x


