import torch
import torch.nn as nn
import torchvision.models as models
from transformers import RobertaModel

class TextEncoder(nn.Module):
    """
    Text encoder using pre-trained RoBERTa.
    Outputs a feature vector of dimension `out_dim` (default 768).
    """
    def __init__(self, pretrained_model_name='roberta-base', out_dim=768):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.roberta.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the representation of the [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)


class RGBEncoder(nn.Module):
    """
    RGB visual encoder using pre-trained ResNet-50.
    Removes the final classification layer and projects to `out_dim`.
    """
    def __init__(self, out_dim=768, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Retain everything except the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x):
        # x expected shape: (batch_size, 3, 224, 224)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DepthEncoder(nn.Module):
    """
    Depth spatial encoder using an AlexNet-BN architecture.
    Crucially, this is NOT pre-trained on generic ImageNet features.
    It is trained from scratch specifically on depth maps to preserve
    local spatial constraints and geometries without color bias.
    """
    def __init__(self, in_channels=1, out_dim=768):
        super().__init__()
        # Note: input channels = 1 for grayscale depth maps.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, out_dim),
        )

        # Initialize weights from scratch using Kaiming normal
        self._initialize_weights()

    def forward(self, x):
        # x expected shape: (batch_size, 1, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
