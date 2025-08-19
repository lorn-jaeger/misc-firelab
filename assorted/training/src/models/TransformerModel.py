import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoImageProcessor, AutoModelForImageClassification

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.transformer = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        # Use ResNet18 to extract visual features
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Flatten the spatial dimensions
        batch_size, _, height, width = x.shape
        x = x.view(batch_size, -1)

        # Concatenate the day of the year feature
        x = torch.cat((x, doys), dim=1)

        # Process the input through the ViT model
        pixel_values = self.image_processor(images=x, return_tensors="pt").pixel_values
        transformer_output = self.transformer(pixel_values).pooler_output
        
        # Classify the output
        output = self.classifier(transformer_output)

        return output