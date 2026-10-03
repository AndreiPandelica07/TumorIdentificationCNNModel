import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,input_height, input_width, num_filters, layer_count,kernel_size, pool_size, in_channels, fc_layer_neuronc):   
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding =kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(input_height // (pool_size ** layer_count) * input_width // (pool_size ** layer_count) * num_filters, fc_layer_neuronc),
            nn.ReLU(),
            nn.Linear(fc_layer_neuronc, 4)  
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layer(x)
        return x
