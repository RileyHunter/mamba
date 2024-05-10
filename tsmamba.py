from einops import rearrange
from mamba_ssm import Mamba
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, forecast, lookback, dropout=0.5, device="cpu"):
        super(Model, self).__init__()
        self.device=device
        self.mamba = Mamba(d_model=d_model, d_state=d_state,d_conv=d_conv, expand=expand).to(device)
        self.d1_nn = nn.Dropout(p=dropout).to(device)
        self.fc1=nn.Linear(in_features=lookback*d_model, out_features=forecast).to(device)

    def forward(self, input):
        bs = input.shape[0]
        h_out = self.mamba(input)
        h_out = rearrange(h_out, 'b l c -> b (l c)')
        h_out = self.d1_nn(h_out)

        out = self.fc1(h_out)
        return out

    def predict(self,input):
        with torch.no_grad():
            predictions=self.forward(input)
        return predictions