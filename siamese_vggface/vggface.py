import torch
import torch.nn as nn
from models import ResNet50, ResNet50VGGFace


class VGGFace(nn.Module):

    def __init__(self, desc='resnet50'):
        super(VGGFace, self).__init__()
        self.desc = desc
        # init descriptor, by default - ResNet50, which produces 8631-dim embedding vector
        if desc == "resnet50":
            self.descriptor = ResNet50()
            self.desc_out_shape = 2048

            # TODO: replace this path somewhere, possibly as class attribute, or some config
            # weights_path = "weights/resnet50_ft_dag.pth"
            weights_path = "weights/resnet50_scratch_dag.pth"

            # load weights, pretrained on VGGFace2 dataset
            state_dict = torch.load(weights_path)
            self.descriptor.load_state_dict(state_dict)

            # freeze pretrained descriptor, as it should not be updated during further training
            for param in self.descriptor.parameters():
                param.requires_grad = False

            # change last layer of the model, so it will return not 8631d embiddings, but 512d
            # num_ftrs = self.descriptor.classifier.in_channels
            # self.descriptor.classifier = nn.Linear(num_ftrs, 512)

        elif desc == "resnet_vggface":
            self.descriptor = ResNet50VGGFace()
            self.desc_out_shape = 2622

            weights_path = "weights/vgg_face_dag.pth"

            # load weights, pretrained on VGGFace2 dataset
            state_dict = torch.load(weights_path)
            self.descriptor.load_state_dict(state_dict)

            # freeze pretrained descriptor, as it should not be updated during further training
            for param in self.descriptor.parameters():
                param.requires_grad = False

        # create two FC layers and dropout
        # TODO: incapsulate number of combinations (3) into variable
        self.fc1 = nn.Linear(self.desc_out_shape * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        # get embedding, by passing images through descriptor
        # TODO: decide which exact layer of the descriptor take as a feature embedding

        if self.desc == "resnet50":
            x1_emb = self.descriptor(x1)[1]
            x2_emb = self.descriptor(x2)[1]
        elif self.desc == "resnet_vggface":
            x1_emb = self.descriptor(x1)
            x2_emb = self.descriptor(x2)

        # TODO: write more possible combinations, possibly encapsulate them into separate functions
        # TODO: replace these combinations with same, as in the paper https://arxiv.org/pdf/2006.00143.pdf
        # different combinations of embeddings
        # x^2 - y^2
        comb_1 = torch.sub(x1_emb.pow(2), x2_emb.pow(2))

        # (x - y)^2
        comb_2 = torch.sub(x1_emb, x2_emb).pow(2)

        # x * y
        comb_3 = x1_emb * x2_emb

        # # concatenate all combinations into 1 vector
        x = torch.cat([comb_1, comb_2, comb_3], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
