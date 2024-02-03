import torch.nn as nn
import torchvision.models as models
from Resnet18 import *
    
class AlzheimerTriMatterNet(nn.Module):
    def __init__(self):
        super(AlzheimerTriMatterNet, self).__init__()
        self.numclass = 4
        self.whitematter_resnet18_model = ResNet18(img_channels=3, num_layers=18, block=BasicBlock, num_classes=4)
        self.graymatter_resnet18_model = ResNet18(img_channels=3, num_layers=18, block=BasicBlock, num_classes=4)
        self.resnet18_model = ResNet18(img_channels=3, num_layers=18, block=BasicBlock, num_classes=4)
        self.global_classification_head = nn.Sequential(
                                                            nn.Linear(512*3,self.numclass),
                                                            nn.Softmax(dim=1),
        )
        
    def forward(self, whitematter, graymatter, original):
        white_output = self.whitematter_resnet18_model(whitematter)
        gray_output = self.graymatter_resnet18_model(graymatter)
        origin_output = self.resnet18_model(original)
        combined_tensor = torch.cat(( white_output, gray_output, origin_output), dim=1)
        output = self.global_classification_head(combined_tensor)
        return output
    