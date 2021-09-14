import torch.nn as nn
from models.baseline import BaselineModel


class SiameseNetwork(nn.Module):
    def __init__(self, descriptor=False):
        super(SiameseNetwork, self).__init__()
        self.descriptor = descriptor


        if descriptor == "arcface":
            # TODO: implement initialization of the ArcFace feature extractor
            pass

        elif descriptor == "vggface":
            # TODO: implement initialization of the VGGFace feature extractor
            pass
        elif descriptor == False:
            self.model = BaselineModel(x1_in_features=2048, x2_in_features=2048)
        else:
            print("There is no such feature extractor. Please select one of [arcface, vggface].")


        # TODO: add definition of the input sizes depending of the feature extractors outputs


    def forward(self, x1, x2):

        # in case we are not using feature extractor
        if not self.descriptor:
            out = self.model(x1, x2)
        else:
            x1 = self.descriptor(x1)
            x2 = self.descriptor(x2)
            out = self.model(x1, x2)

        return out
