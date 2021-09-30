import torch.nn as nn
import torchvision.models as models
class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # pretrained model
        self.network = models.resnet50(pretrained=False)
        self.network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        for param in self.network.fc.parameters():
            param.require_grad = False
        
        num_features = self.network.fc.in_features  # get number of features of last layer
        # -----------------------------------
        self.network.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        return self.network(x)
