import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseCNN(nn.Module):
    def __init__(self, feature_maps, kernel_size, spp_levels, out_dim):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=1, out_channels=feature_maps, kernel_size=kernel_size)
        self.spp = SPP(spp_levels)
        in_dim = self.spp.out_size() * feature_maps
        self.linear = nn.Linear(in_dim, out_dim) # authors don't mention out_dim 
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.spp(x)
        x = self.linear(x)
        return x
    
    
class SPP(nn.Module):
    def __init__(self, levels):
        """"
        levels (tuple): sizes of each pyramid pooling. e.g. (4, 2, 1)
        """
        super().__init__()
        self.levels = levels
        
    def out_size(self):
        return sum(l**2 for l in self.levels)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C * sum(level**2))
        """
        result = [F.adaptive_max_pool2d(x, (l, l)).flatten(start_dim=1)
                  for l in self.levels]
        return torch.cat(result, dim=-1)
        
        
        
        
        
        