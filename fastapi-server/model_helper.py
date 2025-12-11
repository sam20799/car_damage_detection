from PIL import Image
from pydantic_core.core_schema import none_schema
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models


trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

class carClassifierCNNResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # freezing all layers except fc layer
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreezing layer4 we need to train it
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = carClassifierCNNResNet()    #model class
        trained_model.load_state_dict(torch.load("model/saved_model.pth",weights_only=False))  #loaded our saved model weights in model class
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _,predicted = torch.max(output,1)

        return class_names[predicted.item()]
