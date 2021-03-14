import torch
import torch.nn as nn
from models import ResNet50

class VGGFace(nn.Module):

    def __init__(self, desc='resnet50', desc_out_shape=8631, neurons_num=[2048, 512]):
        super().__init__()

        # init descriptor, by default - ResNet50, which produces 8631-dim embedding vector
        if desc=="resnet50":
            self.descriptor = ResNet50()
            assert desc_out_shape == 8631

            # TODO: replace this path somewhere, possibly as class attribute, or some config
            weights_path = "weights/resnet50_ft_dag.pth"

            # load weights, pretrained on VGGFace2 dataset
            state_dict = torch.load(weights_path)
            self.descriptor.load_state_dict(state_dict)

            # freeze pretrained descriptor, as it should not be updated during further training
            for param in self.descriptor.parameters():
                param.requires_grad = False


        # create two FC layers and dropout
        # TODO: incapsulate number of combinations (3) into variable
        self.fc1 = nn.Linear(desc_out_shape * 3, neurons_num[0])
        self.fc2 = nn.Linear(neurons_num[0], neurons_num[1])
        self.fc3 = nn.Linear(neurons_num[1], 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x1, x2):
        # get embedding, by passing images through descriptor
        x1_emb = self.descriptor(x1)[0]
        x2_emb = self.descriptor(x2)[0]

        # TODO: write more possible combinations, possibly encapsulate them into separate functions
        # TODO: replace these combinations with same, as in the paper https://arxiv.org/pdf/2006.00143.pdf
        # different combinations of embeddings
        # x^2 - y^2
        comb_1 = torch.sub(x1_emb.pow(2), x2_emb.pow(2))

        # (x - y)^2
        comb_2 = torch.sub(x1_emb, x2_emb).pow(2)

        # x * y
        comb_3 = x1_emb * x2_emb


        # concatenate all combinations into 1 vector
        x = torch.cat([comb_1, comb_2, comb_3], dim=1)


        # pass through FC layers resulting combination
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x





