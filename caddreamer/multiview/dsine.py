import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_pixel_cords(w: float, h: float) -> np.typing.NDArray[np.float32]:
    pixel_cords = np.ones((w, h, 3), dtype=np.float32)
    x_range = np.concatenate([np.arange(w).reshape(1, w) * h], axis=0)
    y_range = np.concatenate([np.arange(h).reshape(h, 1) * w], axis=1)

    pixel_cords[:, :, 0] = x_range.T
    pixel_cords[:, :, 1] = y_range.T

    return torch.from_numpy(pixel_cords).unsqueeze(0)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, output_dim, ks=3):
        super(ConvGRU, self).__init__()
        p = (ks - 1) // 2
        self.conv1 = nn.Conv2d(hidden_dim + output_dim, hidden_dim*2, ks, padding=p)
        self.conv2 = nn.Conv2d(hidden_dim + output_dim, hidden_dim*2, ks, padding=p)
        self.conv3 = nn.Conv2d(hidden_dim + output_dim, hidden_dim*2, ks, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.conv1(hx))
        r = torch.sigmoid(self.conv2(hx))
        q = torch.tanh(self.conv3(torch.cat([r*h, x])), dim=1)
        h = (1-z)*h + z*q
        return h

