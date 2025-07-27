import torch
import torch.nn as nn
#################################
from RiverModel.unet.unet_model import UNet as RiverNet  # unet

device = torch.device('cuda')
class RiverLoss(nn.Module):
    def __init__(self, device):
        super(RiverLoss, self).__init__()
        self.device = device
    def forward(self, input, targets):

        fake_river_heatmap = unet(input.to(self.device))
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        loss = criterion(fake_river_heatmap, targets)

        return loss


extract_river_Weights = './RiverModel/unet/unet.pth'
unet = RiverNet(n_channels=1, n_classes=1)  # unet
unet.load_state_dict(torch.load(extract_river_Weights))
unet.to(device)
river_conterion = RiverLoss(device)
river_conterion.to(device)
