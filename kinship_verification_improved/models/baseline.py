import torch.nn as nn
import torch


class fc_block(nn.Module):
    def __init__(self, in_features, out_features=1000, d=0.5):
        super(fc_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(d)
        )

    def forward(self, x):
        return self.block(x)


class fc_identity(nn.Module):
    def __init__(self, in_features, out_features=1000, d=0.5):
        super(fc_identity, self).__init__()
        self.block = nn.Sequential(
            fc_block(in_features, out_features, d),
            nn.Linear(in_features=out_features, out_features=in_features),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.block(x)
        y = torch.stack([x, y]).sum(0)
        y = self.relu(y)
        return y


class BaselineModel(nn.Module):
    def __init__(self, x1_in_features, x2_in_features, n=2048, out_channels=1, *args, **kwargs):
        super(BaselineModel, self).__init__()


        self.x1_branch = nn.Sequential(
            fc_block(x1_in_features, n, d=0.2),
            fc_identity(n, n, d=0.2),
        )

        self.x2_branch = nn.Sequential(
            fc_block(x2_in_features, n, d=0.2),
            fc_identity(n, n, d=0.2),
        )

        self.cat = nn.Sequential(
            fc_identity(n*2, n*2, d=0.2),
            fc_block(n*2, n, d=0.2),
            nn.Linear(n, out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.x1_branch(x1)
        x2 = self.x2_branch(x2)
        cat = torch.cat([x1, x2], dim=1)
        cat = self.cat(cat)
        return cat

