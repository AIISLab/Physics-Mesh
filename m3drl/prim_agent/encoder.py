import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimEncoder(nn.Module):
    def __init__(self, prim_params, steps, shape_ref):
        super(PrimEncoder, self).__init__()
        self.shape = shape_ref

        # Encoder for shape reference
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.shape[0], out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # FC Layers with ReLU activation for primitive params
        self.fc = nn.Sequential(
            nn.Linear(in_features=prim_params, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU()
        )

        # Final FC Layer with ReLU activation for step indicator
        self.step_fc = nn.Sequential(
            nn.Linear(in_features=steps, out_features=256),
            nn.ReLU()
        )

        # Concatenated streams with 3 FC layers, 2 with ReLU activation and 1 to output Q-values for actions
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=256 + 256 + 64 * (self.shape[1] // 8), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        ),

    def forward(self, shape_ref: torch.Tensor, prim_params: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        x = self.encoder(shape_ref)
        x = torch.flatten(x, start_dim=1)

        prim_params = self.fc(prim_params)
        step = self.step_fc(step)

        x = torch.cat((x, prim_params, step), dim=1)
        x = self.final_fc(x)

        return x[0]